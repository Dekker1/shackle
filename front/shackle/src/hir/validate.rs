//! Final validation step for HIR representation.
//!
//! This module contains miscellaneous validation steps which require the whole
//! program HIR, and can't be done on a per-item basis.
//!
//! - Check for illegal overloading/duplicate definitions
//! - Check for multiple definitions of variables
//! - Check for multiple solve items

use std::{collections::hash_map::Entry, sync::Arc};

use rustc_hash::FxHashMap;

use crate::{
	error::{
		AdditionalSolveItem, DuplicateAssignment, DuplicateFunction, FunctionAlreadyDefined,
		MultipleAssignments, MultipleSolveItems,
	},
	hir::ids::{ItemRef, NodeRef},
	ty::{FunctionEntry, OverloadingError},
	Error,
};

use super::{
	db::Hir,
	ids::{EntityRef, LocalItemRef},
	PatternTy,
};

/// Validate HIR
pub fn validate_hir(db: &dyn Hir) -> Arc<Vec<Error>> {
	let mut diagnostics = Vec::new();
	// Validate overloading
	let global_scope = db.lookup_global_scope();
	for (_, ps) in global_scope.functions(0) {
		let mut overloads = Vec::new();
		for p in ps.iter() {
			let signature = db.lookup_item_signature(p.item());
			match &signature.patterns[p] {
				PatternTy::Function(f) => overloads.push((*p, *f.clone())),
				PatternTy::EnumConstructor(fs) => {
					overloads.extend(fs.iter().map(|f| (*p, f.clone())))
				}
				_ => unreachable!(),
			}
		}
		let errors = FunctionEntry::check_overloading(db.upcast(), overloads);
		diagnostics.extend(errors.iter().map(|e| match e {
			OverloadingError::FunctionAlreadyDefined {
				first: (first_pat, first_fn),
				others,
			} => {
				let item = first_pat.item();
				let model = item.model(db);
				let data = item.local_item_ref(db).data(&*model);
				let name = data[first_pat.pattern()].identifier().unwrap();
				let signature = first_fn.overload.pretty_print_item(db.upcast(), name);
				let (src, span) = NodeRef::from(first_pat.into_entity(db)).source_span(db);
				FunctionAlreadyDefined {
					src,
					span,
					signature,
					others: others
						.iter()
						.map(|(p, _)| {
							let (src, span) = NodeRef::from(p.into_entity(db)).source_span(db);
							DuplicateFunction { src, span }
						})
						.collect(),
				}
				.into()
			}
		}))
	}

	// Check for multiple assignments to variables
	let mut assignments: FxHashMap<_, Vec<NodeRef>> = FxHashMap::default();
	for m in db.resolve_includes().unwrap().iter() {
		let model = db.lookup_model(*m);
		for (i, a) in model.assignments.iter() {
			let item_ref = ItemRef::new(db, *m, i);
			let types = db.lookup_item_types(item_ref);
			if let Some(p) = types.name_resolution(a.assignee) {
				match assignments.entry(p) {
					Entry::Occupied(mut e) => {
						e.get_mut().push(item_ref.into());
					}
					Entry::Vacant(e) => {
						let mut v = Vec::new();
						let resolved_item = p.item();
						match resolved_item.local_item_ref(db) {
							LocalItemRef::Declaration(d) => {
								let model = resolved_item.model(db);
								if let Some(def) = model[d].definition {
									v.push(EntityRef::new(db, resolved_item, def).into());
								}
							}
							_ => (),
						}
						v.push(item_ref.into());
						e.insert(v);
					}
				}
			}
		}
		for (i, a) in model.enum_assignments.iter() {
			let item_ref = ItemRef::new(db, *m, i);
			let types = db.lookup_item_types(item_ref);
			if let Some(p) = types.name_resolution(a.assignee) {
				match assignments.entry(p) {
					Entry::Occupied(mut e) => {
						e.get_mut().push(item_ref.into());
					}
					Entry::Vacant(e) => {
						let mut v = Vec::new();
						let resolved_item = p.item();
						match resolved_item.local_item_ref(db) {
							LocalItemRef::Enumeration(e) => {
								let model = resolved_item.model(db);
								if model[e].definition.is_some() {
									v.push(p.into_entity(db).into());
								}
							}
							_ => (),
						}
						v.push(item_ref.into());
						e.insert(v);
					}
				}
			}
		}
	}
	for (p, asgs) in assignments {
		if asgs.len() > 1 {
			let model = p.item().model(db);
			let variable = p.item().local_item_ref(db).data(&*model)[p.pattern()]
				.identifier()
				.unwrap()
				.pretty_print(db);
			let mut asgs = asgs.into_iter();
			let (src, span) = asgs.next().unwrap().source_span(db);
			let others = asgs
				.map(|i| {
					let (src, span) = i.source_span(db);
					DuplicateAssignment { src, span }
				})
				.collect();
			diagnostics.push(
				MultipleAssignments {
					src,
					span,
					variable,
					others,
				}
				.into(),
			)
		}
	}

	// Check for multiple solve items
	let mut solve_items = Vec::new();
	for m in db.resolve_includes().unwrap().iter() {
		let model = db.lookup_model(*m);
		for (i, _) in model.solves.iter() {
			let item_ref = ItemRef::new(db, *m, i);
			solve_items.push(item_ref);
		}
	}
	if solve_items.len() > 1 {
		let mut iter = solve_items.into_iter();
		let first = iter.next().unwrap();
		let (src, span) = NodeRef::from(first).source_span(db);
		diagnostics.push(
			MultipleSolveItems {
				src,
				span,
				others: iter
					.map(|i| {
						let (src, span) = NodeRef::from(i).source_span(db);
						AdditionalSolveItem { src, span }
					})
					.collect(),
			}
			.into(),
		);
	}
	Arc::new(diagnostics)
}
