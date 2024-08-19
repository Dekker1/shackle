use crate::{
	constants::IdentifierRegistry,
	file::ModelRef,
	hir::{db::Hir, source::SourceMap, *},
	syntax::xcsp::{XcspArray, XcspConstraint, XcspObjectives, XcspVariable},
	Error,
};

/// Collects AST items into an HIR model
pub struct XcspItemCollector<'a> {
	db: &'a dyn Hir,
	identifiers: &'a IdentifierRegistry,
	model: Model,
	source_map: SourceMap,
	diagnostics: Vec<Error>,
	owner: ModelRef,
}

impl<'a> XcspItemCollector<'a> {
	/// Create a new item collector
	pub fn new(db: &'a dyn Hir, identifiers: &'a IdentifierRegistry, owner: ModelRef) -> Self {
		Self {
			db,
			identifiers,
			model: Model::default(),
			source_map: SourceMap::default(),
			diagnostics: Vec::new(),
			owner,
		}
	}

	/// Lower a variable to HIR
	pub fn collect_variable(&mut self, _item: XcspVariable) {
		todo!()
	}

	/// Lower an array to HIR
	pub fn collect_array(&mut self, _item: XcspArray) {
		todo!()
	}

	/// Lower an constraint to HIR
	pub fn collect_constraint(&mut self, _item: XcspConstraint) {
		todo!()
	}

	/// Lower the objectives to HIR
	pub fn collect_objectives(&mut self, _item: XcspObjectives) {
		todo!()
	}

	/// Finish lowering
	pub fn finish(self) -> (Model, SourceMap, Vec<Error>) {
		(self.model, self.source_map, self.diagnostics)
	}
}
