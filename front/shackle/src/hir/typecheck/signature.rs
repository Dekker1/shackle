/// Types of signatures - the type required when referring to an item
///
/// E.g.
/// - Function parameter/return type
/// - Variable declaration LHS types
use rustc_hash::FxHashMap;

use crate::{
	error::{SyntaxError, TypeInferenceFailure, TypeMismatch},
	hir::{
		db::Hir,
		ids::{EntityRef, ExpressionRef, ItemRef, LocalItemRef, NodeRef, PatternRef},
		EnumerationCase, Goal, IdentifierRegistry, ItemData, Pattern, Type,
	},
	ty::{
		EnumRef, FunctionEntry, FunctionType, OptType, OverloadedFunction, PolymorphicFunctionType,
		Ty, TyData, TyVar, TyVarRef, TypeRegistry, VarType,
	},
	Error,
};

use super::{PatternTy, TypeContext, Typer};

/// Collected types for an item signature
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureTypes {
	/// Types of declarations
	pub patterns: FxHashMap<PatternRef, PatternTy>,
	/// Types of expressions
	pub expressions: FxHashMap<ExpressionRef, Ty>,
	/// Identifier resolution
	pub identifier_resolution: FxHashMap<ExpressionRef, PatternRef>,
	/// Pattern resolution
	pub pattern_resolution: FxHashMap<PatternRef, PatternRef>,
}

/// Context for typing an item signature
pub struct SignatureTypeContext {
	starting_item: ItemRef,
	data: SignatureTypes,
	diagnostics: Vec<Error>,
}

impl SignatureTypeContext {
	/// Create a new signature type context
	pub fn new(item: ItemRef) -> Self {
		Self {
			starting_item: item,
			data: SignatureTypes {
				patterns: FxHashMap::default(),
				expressions: FxHashMap::default(),
				identifier_resolution: FxHashMap::default(),
				pattern_resolution: FxHashMap::default(),
			},
			diagnostics: Vec::new(),
		}
	}

	/// Compute the signature of the given item
	pub fn type_item(
		&mut self,
		db: &dyn Hir,
		types: &TypeRegistry,
		identifiers: &IdentifierRegistry,
		item: ItemRef,
	) {
		let model = &*item.model(db);
		let it = item.local_item_ref(db);
		let data = it.data(model);
		match it {
			LocalItemRef::Function(f) => {
				let it = &model[f];
				// Set as computing so if there's a call to a function with this name we can break the cycle
				// (since if the call is actually not referring to this overload, it should work)
				self.add_declaration(PatternRef::new(item, it.pattern), PatternTy::Computing);
				let ty_params = it
					.type_inst_vars
					.iter()
					.map(|tv| {
						let ty_var = TyVarRef::new(db, PatternRef::new(item, tv.name));
						let type_var = TyVar {
							ty_var,
							varifiable: tv.is_varifiable,
							enumerable: tv.is_enum,
							indexable: tv.is_indexable,
						};
						self.add_declaration(
							PatternRef::new(item, tv.name),
							PatternTy::TyVar(type_var),
						);
						ty_var
					})
					.chain(
						it.parameters
							.iter()
							.flat_map(|p| Type::anonymous_ty_vars(p.declared_type, &it.data))
							.map(|t| match &it.data[t] {
								Type::AnonymousTypeInstVar { pattern, .. } => {
									TyVarRef::new(db, PatternRef::new(item, *pattern))
								}
								_ => unreachable!(),
							}),
					)
					.collect::<Box<[_]>>();
				let params = it
					.parameters
					.iter()
					.enumerate()
					.map(|(i, p)| {
						let mut had_error = false;
						let annotated_expression = p
							.annotations
							.iter()
							.find(|ann| match &it.data[**ann] {
								crate::hir::Expression::Identifier(i) => {
									i.lookup(db) == "annotated_expression"
								}
								_ => false,
							})
							.copied();
						if i > 0 {
							if let Some(ann) = annotated_expression {
								let (src, span) =
									NodeRef::from(EntityRef::new(db, item, ann)).source_span(db);
								self.add_diagnostic(
									item,
									SyntaxError {
										src,
										span,
										msg: "'annotated_expression' only allowed on first function parameter.".to_owned(),
										other: Vec::new(),
									},
								);
							}
						}
						for t in Type::any_types(p.declared_type, &it.data) {
							let (src, span) =
								NodeRef::from(EntityRef::new(db, item, t)).source_span(db);
							self.add_diagnostic(
								item,
								TypeInferenceFailure {
									src,
									span,
									msg: "Incomplete parameter types are not allowed".to_owned(),
								},
							);
							had_error = true;
						}
						let ty = if had_error {
							types.error
						} else {
							Typer::new(db, types, identifiers, self, item, data)
								.complete_type(p.declared_type, None)
						};
						if let Some(pat) = p.pattern {
							self.add_declaration(
								PatternRef::new(item, pat),
								PatternTy::Variable(ty),
							);
						}
						ty
					})
					.collect();
				let pattern = PatternRef::new(item, it.pattern);
				if ty_params.is_empty() {
					let f = FunctionType {
						return_type: types.error,
						params,
					};
					self.add_declaration(
						pattern,
						PatternTy::Function(Box::new(FunctionEntry {
							has_body: it.body.is_some(),
							overload: OverloadedFunction::Function(f),
						})),
					);
				} else {
					let p = PolymorphicFunctionType {
						return_type: types.error,
						ty_params,
						params,
					};
					self.add_declaration(
						pattern,
						PatternTy::Function(Box::new(FunctionEntry {
							has_body: it.body.is_some(),
							overload: OverloadedFunction::PolymorphicFunction(p),
						})),
					);
				}

				let mut had_error = false;
				for t in Type::any_types(it.return_type, &it.data)
					.chain(Type::anonymous_ty_vars(it.return_type, &it.data))
				{
					let (src, span) = NodeRef::from(EntityRef::new(db, item, t)).source_span(db);
					self.add_diagnostic(
						item,
						TypeInferenceFailure {
							src,
							span,
							msg: "Incomplete return type not allowed".to_owned(),
						},
					);
					had_error = true;
				}
				let return_type = if had_error {
					types.error
				} else {
					Typer::new(db, types, identifiers, self, item, data)
						.complete_type(it.return_type, None)
				};

				let d = self.data.patterns.get_mut(&pattern).unwrap();
				match d {
					PatternTy::Function(function) => match function.as_mut() {
						FunctionEntry {
							overload: OverloadedFunction::Function(f),
							..
						} => {
							f.return_type = return_type;
						}
						FunctionEntry {
							overload: OverloadedFunction::PolymorphicFunction(p),
							..
						} => {
							p.return_type = return_type;
						}
					},
					_ => unreachable!(),
				}
			}
			LocalItemRef::Declaration(d) => {
				let it = &model[d];
				for p in Pattern::identifiers(it.pattern, data) {
					self.add_declaration(PatternRef::new(item, p), PatternTy::Computing);
				}
				let mut typer = Typer::new(db, types, identifiers, self, item, data);
				if data[it.declared_type].is_complete(data) {
					// Use LHS type only
					let expected = typer.complete_type(it.declared_type, None);
					typer.collect_pattern(None, it.pattern, expected);
					drop(typer);
				} else {
					typer.collect_declaration(it);
				}
			}
			LocalItemRef::Enumeration(e) => {
				let it = &model[e];
				let ty = Ty::par_enum(
					db.upcast(),
					EnumRef::new(db, PatternRef::new(item, it.pattern)),
				);
				self.add_declaration(
					PatternRef::new(item, it.pattern),
					PatternTy::Variable(Ty::par_set(db.upcast(), ty).unwrap()),
				);
				if let Some(cases) = &it.definition {
					self.add_enum_cases(db, types, identifiers, item, data, ty, cases);
				}
			}
			LocalItemRef::EnumAssignment(e) => {
				let it = &model[e];
				let set_ty = Typer::new(db, types, identifiers, self, item, data)
					.collect_expression(it.assignee, None);
				let ty = match set_ty.lookup(db.upcast()) {
					TyData::Set(_, _, e) => e,
					_ => unreachable!(),
				};
				self.add_enum_cases(db, types, identifiers, item, data, ty, &it.definition);
			}
			LocalItemRef::Solve(s) => {
				let it = &model[s];
				match &it.goal {
					Goal::Maximize { pattern, objective }
					| Goal::Minimize { pattern, objective } => {
						self.add_declaration(PatternRef::new(item, *pattern), PatternTy::Computing);
						let actual = Typer::new(db, types, identifiers, self, item, data)
							.collect_expression(*objective, None);
						if !actual.is_subtype_of(db.upcast(), types.var_float) {
							let (src, span) =
								NodeRef::from(EntityRef::new(db, item, *objective)).source_span(db);
							self.add_diagnostic(
								item,
								TypeMismatch {
									src,
									span,
									msg: format!(
										"Objective must be numeric, but got '{}'",
										actual.pretty_print(db.upcast())
									),
								},
							);
						}
						self.add_declaration(
							PatternRef::new(item, *pattern),
							PatternTy::Variable(actual),
						);
					}
					_ => (),
				}
			}
			LocalItemRef::TypeAlias(t) => {
				let it = &model[t];
				let pat = PatternRef::new(item, it.name);
				self.add_declaration(pat, PatternTy::Computing);
				let ty = Typer::new(db, types, identifiers, self, item, data)
					.complete_type(it.aliased_type, None);
				self.add_declaration(pat, PatternTy::TypeAlias(ty));
			}
			_ => unreachable!("Item {:?} does not have signature", it),
		}
	}

	fn add_enum_cases(
		&mut self,
		db: &dyn Hir,
		types: &TypeRegistry,
		identifiers: &IdentifierRegistry,
		item: ItemRef,
		data: &ItemData,
		ty: Ty,
		cases: &[EnumerationCase],
	) {
		for case in cases.iter() {
			if case.parameters.is_empty() {
				self.add_declaration(PatternRef::new(item, case.pattern), PatternTy::EnumAtom(ty));
			} else {
				let param_types = {
					let mut typer = Typer::new(db, types, identifiers, self, item, data);
					case.parameters
						.iter()
						.map(|p| typer.complete_type(*p, None))
						.collect::<Box<[_]>>()
				};

				let mut had_error = false;
				for (p, t) in case.parameters.iter().zip(param_types.iter()) {
					if t.contains_error(db.upcast()) {
						had_error = true;
					}
					if !t.known_par(db.upcast()) || !t.known_enumerable(db.upcast()) {
						let (src, span) =
							NodeRef::from(EntityRef::new(db, item, *p)).source_span(db);
						self.add_diagnostic(
							item,
							TypeMismatch {
								src,
								span,
								msg: format!(
									"Expected par enumerable constructor parameter, but got '{}'",
									t.pretty_print(db.upcast())
								),
							},
						);
						had_error = true;
					}

					for unbounded in Type::primitives(*p, data) {
						let (src, span) =
							NodeRef::from(EntityRef::new(db, item, unbounded)).source_span(db);
						self.add_diagnostic(
							item,
							TypeInferenceFailure {
								src,
								span,
								msg: "Unbounded enum constructor parameters not supported"
									.to_owned(),
							},
						);
					}
				}

				let mut constructors = Vec::new();
				// C(a, b, ..) -> E
				constructors.push(FunctionType {
					return_type: ty,
					params: param_types.clone(),
				});
				if !had_error {
					// C(opt a, opt b, ..) -> opt E
					constructors.push(FunctionType {
						return_type: ty.with_opt(db.upcast(), OptType::Opt),
						params: param_types
							.iter()
							.map(|t| t.with_opt(db.upcast(), OptType::Opt))
							.collect(),
					});
					// C(var a, var b, ..) -> var E
					constructors.push(FunctionType {
						return_type: ty.with_inst(db.upcast(), VarType::Var).unwrap(),
						params: param_types
							.iter()
							.map(|t| t.with_inst(db.upcast(), VarType::Var).unwrap())
							.collect(),
					});
					// C(var opt a, var opt b, ..) -> var opt E
					constructors.push(FunctionType {
						return_type: ty
							.with_inst(db.upcast(), VarType::Var)
							.unwrap()
							.with_opt(db.upcast(), OptType::Opt),
						params: param_types
							.iter()
							.map(|t| {
								t.with_inst(db.upcast(), VarType::Var)
									.unwrap()
									.with_opt(db.upcast(), OptType::Opt)
							})
							.collect(),
					});
					// C(set of a, set of b, ..) -> set of E
					constructors.push(FunctionType {
						return_type: Ty::par_set(db.upcast(), ty).unwrap(),
						params: param_types
							.iter()
							.map(|t| Ty::par_set(db.upcast(), *t).unwrap())
							.collect(),
					});
					// C(var set of a, var set of b, ..) -> var set of E
					constructors.push(FunctionType {
						return_type: Ty::par_set(db.upcast(), ty)
							.unwrap()
							.with_inst(db.upcast(), VarType::Var)
							.unwrap(),
						params: param_types
							.iter()
							.map(|t| {
								Ty::par_set(db.upcast(), *t)
									.unwrap()
									.with_inst(db.upcast(), VarType::Var)
									.unwrap()
							})
							.collect(),
					});
				}

				self.add_declaration(
					PatternRef::new(item, case.pattern),
					PatternTy::EnumConstructor(
						constructors
							.into_iter()
							.map(|f| FunctionEntry {
								has_body: true,
								overload: OverloadedFunction::Function(f),
							})
							.collect(),
					),
				);
			}
		}
	}

	/// Get results of typing
	pub fn finish(self) -> (SignatureTypes, Vec<Error>) {
		(self.data, self.diagnostics)
	}
}

impl TypeContext for SignatureTypeContext {
	fn add_declaration(&mut self, pattern: PatternRef, declaration: PatternTy) {
		let old = self.data.patterns.insert(pattern, declaration);
		assert!(
			matches!(old, None | Some(PatternTy::Computing)),
			"Tried to add declaration for {:?} twice",
			pattern
		);
	}
	fn add_expression(&mut self, expression: ExpressionRef, ty: Ty) {
		let old = self.data.expressions.insert(expression, ty);
		assert!(
			old.is_none(),
			"Tried to add type for expression {:?} twice",
			expression
		);
	}
	fn add_identifier_resolution(&mut self, expression: ExpressionRef, resolution: PatternRef) {
		let old = self
			.data
			.identifier_resolution
			.insert(expression, resolution);
		assert!(
			old.is_none(),
			"Tried to add identifier resolution for {:?} twice",
			expression
		);
	}
	fn add_pattern_resolution(&mut self, pattern: PatternRef, resolution: PatternRef) {
		let old = self.data.pattern_resolution.insert(pattern, resolution);
		assert!(
			old.is_none(),
			"Tried to add pattern resolution for {:?} twice",
			pattern
		);
	}
	fn add_diagnostic(&mut self, item: ItemRef, e: impl Into<Error>) {
		// Suppress errors from other items
		if item == self.starting_item {
			self.diagnostics.push(e.into());
		}
	}

	fn type_pattern(
		&mut self,
		db: &dyn Hir,
		types: &TypeRegistry,
		identifiers: &IdentifierRegistry,
		pattern: PatternRef,
	) -> PatternTy {
		// When computing signatures, we always type everything required
		// So other signatures get typed as well
		if let Some(d) = self.data.patterns.get(&pattern).cloned() {
			return d;
		}
		self.type_item(db, types, identifiers, pattern.item());
		self.data.patterns[&pattern].clone()
	}
}
