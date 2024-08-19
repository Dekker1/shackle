//! Functionality for converting AST nodes to HIR nodes
//! for the respective modelling languages.

pub mod eprime;
pub mod minizinc;
pub mod xcsp3;

#[cfg(test)]
pub mod test;

use std::sync::Arc;

use self::{eprime::ItemCollector as EPrimeItemCollector, minizinc::ItemCollector};
use crate::{
	constants::IdentifierRegistry,
	file::ModelRef,
	hir::{db::Hir, lower::xcsp3::XcspItemCollector, source::SourceMap, *},
	syntax::ast::ConstraintModel,
	Error,
};

/// Lower a model to HIR
pub fn lower_items(db: &dyn Hir, model: ModelRef) -> (Arc<Model>, Arc<SourceMap>, Arc<Vec<Error>>) {
	let ast = match db.ast(*model) {
		Ok(m) => m,
		Err(e) => return (Default::default(), Default::default(), Arc::new(vec![e])),
	};
	let identifiers = IdentifierRegistry::new(db);
	match ast {
		ConstraintModel::MznModel(ast) => {
			let mut ctx = ItemCollector::new(db, &identifiers, model);
			for item in ast.items() {
				ctx.collect_item(item);
			}
			let (m, sm, e) = ctx.finish();
			(Arc::new(m), Arc::new(sm), Arc::new(e))
		}
		ConstraintModel::EPrimeModel(ast) => {
			let mut ctx = EPrimeItemCollector::new(db, &identifiers, model);
			ctx.preprocess(ast.items());
			for item in ast.items() {
				ctx.collect_item(item);
			}
			ctx.add_solve();
			let (m, sm, e) = ctx.finish();
			(Arc::new(m), Arc::new(sm), Arc::new(e))
		}
		ConstraintModel::XcspModel(instance) => {
			let mut ctx = XcspItemCollector::new(db, &identifiers, model);
			for item in instance.variables {
				ctx.collect_variable(item);
			}
			for item in instance.arrays {
				ctx.collect_array(item);
			}
			for item in instance.constraints {
				ctx.collect_constraint(item);
			}
			ctx.collect_objectives(instance.objectives);
			let (m, sm, e) = ctx.finish();
			(Arc::new(m), Arc::new(sm), Arc::new(e))
		}
	}
}
