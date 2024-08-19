#![allow(missing_docs)]
//! Database queries for syntax parsing

use miette::{ByteOffset, SourceSpan};
use tree_sitter::Parser;

use super::{ast::ConstraintModel, cst::Cst, eprime::EPrimeModel, minizinc::MznModel};
use crate::{
	db::{FileReader, Upcast},
	diagnostics::SyntaxError,
	file::{FileRef, InputLang, SourceFile},
	syntax::xcsp::XcspModel,
	Result,
};

/// Syntax parsing queries
#[salsa::query_group(SourceParserStorage)]
pub trait SourceParser: FileReader + Upcast<dyn FileReader> {
	/// Produce a CST for the given file.
	///
	/// Only gives an `Err` result if getting the file contents failed.
	/// Otherwise, the error is contained in the CST.
	fn cst(&self, file: FileRef) -> Result<Cst>;

	/// Produce an AST for the given file.
	///
	/// Only gives an `Err` result if getting the file contents failed.
	/// Otherwise, the error is contained in the CST.
	fn ast(&self, file: FileRef) -> Result<ConstraintModel>;
}

fn cst(db: &dyn SourceParser, file: FileRef) -> Result<Cst> {
	let contents = file.contents(db.upcast())?;

	let tree_sitter_lang = match file.lang(db.upcast()) {
		InputLang::MiniZinc => tree_sitter_minizinc::language(),
		InputLang::EPrime => tree_sitter_eprime::language(),
		_ => unreachable!("cst should only be called on  Essence' or MiniZinc files"),
	};

	let mut parser = Parser::new();
	parser
		.set_language(&tree_sitter_lang)
		.expect("Failed to set Tree Sitter parser language");
	let tree = parser
		.parse(contents.as_bytes(), None)
		.expect("Tree Sitter parser did not return tree object");

	Ok(Cst::new(tree, file, contents))
}

fn ast(db: &dyn SourceParser, file: FileRef) -> Result<ConstraintModel> {
	match file.lang(db.upcast()) {
		InputLang::MiniZinc => {
			let cst = db.cst(file)?;
			Ok(ConstraintModel::MznModel(MznModel::new(cst)))
		}
		InputLang::EPrime => {
			let cst = db.cst(file)?;
			Ok(ConstraintModel::EPrimeModel(EPrimeModel::new(cst)))
		}
		InputLang::Xcsp3 => {
			let contents = file.contents(db.upcast())?;
			let instance: XcspModel =
				quick_xml::de::from_str(&contents).map_err(|e| SyntaxError {
					src: SourceFile::new(file, db.upcast()),
					span: SourceSpan::new((0 as ByteOffset).into(), contents.len()),
					msg: e.to_string(),
					other: Vec::new(),
				})?;
			Ok(ConstraintModel::XcspModel(instance))
		}
		_ => unreachable!("ast should only be called on model files"),
	}
}
