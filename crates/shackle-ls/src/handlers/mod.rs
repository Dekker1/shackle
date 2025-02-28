mod completions;
mod format;
mod goto_definition;
mod hover;
mod references;
mod rename_symbol;
mod semantic_tokens;
mod vfs;
mod view_ast;
mod view_cst;
mod view_format_ir;
mod view_hir;
mod view_pretty_print;
mod view_scope;

pub use self::{
	completions::*, format::*, goto_definition::*, hover::*, references::*, rename_symbol::*,
	semantic_tokens::*, vfs::*, view_ast::*, view_cst::*, view_format_ir::*, view_hir::*,
	view_pretty_print::*, view_scope::*,
};

#[cfg(test)]
pub mod test {
	use std::{
		ops::Deref,
		panic::RefUnwindSafe,
		path::{Path, PathBuf},
		str::FromStr,
		sync::Arc,
	};

	use expect_test::Expect;
	use lsp_server::ResponseError;
	use shackle_compiler::{
		db::{CompilerDatabase, FileReader, Inputs},
		diagnostics::FileError,
		file::{FileHandler, InputFile, InputLang},
	};

	use crate::{db::LanguageServerContext, dispatch::RequestHandler};

	struct MockFileHandler(String);

	impl FileHandler for MockFileHandler {
		fn durable(&self) -> bool {
			true
		}

		fn read_file(&self, path: &Path) -> Result<Arc<String>, FileError> {
			if path == PathBuf::from_str("test.mzn").unwrap() {
				return Ok(Arc::new(self.0.clone()));
			}
			std::fs::read_to_string(path)
				.map(Arc::new)
				.map_err(|err| FileError {
					file: path.to_path_buf(),
					message: err.to_string(),
					other: Vec::new(),
				})
		}

		fn snapshot(&self) -> Box<dyn FileHandler + RefUnwindSafe> {
			unimplemented!()
		}
	}

	struct MockDatabase {
		db: CompilerDatabase,
		#[allow(dead_code)]
		workspace: Option<lsp_types::Uri>,
	}

	impl Deref for MockDatabase {
		type Target = CompilerDatabase;

		fn deref(&self) -> &Self::Target {
			&self.db
		}
	}

	impl LanguageServerContext for MockDatabase {
		fn set_active_file_from_document(
			&mut self,
			_doc: &lsp_types::TextDocumentIdentifier,
		) -> Result<shackle_compiler::file::ModelRef, lsp_server::ResponseError> {
			Ok(self.input_models()[0])
		}

		fn get_workspace_uri(&self) -> Option<&lsp_types::Uri> {
			self.workspace.as_ref()
		}
	}

	pub fn run_handler<H, R, T>(
		model: &str,
		no_stdlib: bool,
		params: R::Params,
	) -> Result<R::Result, ResponseError>
	where
		H: RequestHandler<R, T>,
		R: lsp_types::request::Request,
	{
		let mut db = MockDatabase {
			db: CompilerDatabase::with_file_handler(Box::new(MockFileHandler(model.to_string()))),
			workspace: lsp_types::Uri::from_str("file:///").ok(),
		};
		db.db.set_ignore_stdlib(no_stdlib);
		db.db.set_input_files(Arc::new(vec![InputFile::Path(
			PathBuf::from_str("test.mzn").unwrap(),
			InputLang::MiniZinc,
		)]));
		H::prepare(&mut db, params).and_then(|t| H::execute(&db, t))
	}

	/// Test an LSP handler
	pub fn test_handler<H, R, T>(model: &str, no_stdlib: bool, params: R::Params, expected: Expect)
	where
		H: RequestHandler<R, T>,
		R: lsp_types::request::Request,
	{
		let actual = run_handler::<H, R, T>(model, no_stdlib, params);
		expected.assert_eq(&serde_json::to_string_pretty(&actual).unwrap());
	}

	/// Test an LSP handler which returns a string
	pub fn test_handler_display<H, R, T>(
		model: &str,
		no_stdlib: bool,
		params: R::Params,
		expected: Expect,
	) where
		H: RequestHandler<R, T>,
		R: lsp_types::request::Request,
		R::Result: std::fmt::Display,
	{
		let actual = run_handler::<H, R, T>(model, no_stdlib, params);
		if let Ok(s) = actual {
			expected.assert_eq(&s.to_string());
		} else {
			expected.assert_eq(&serde_json::to_string_pretty(&actual).unwrap());
		}
	}
}
