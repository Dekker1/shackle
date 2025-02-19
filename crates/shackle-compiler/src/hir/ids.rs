//! IDs for referencing HIR nodes.

use std::sync::Arc;

use miette::SourceSpan;

use super::{
	db::Hir, Annotation, Assignment, Constraint, Declaration, EnumAssignment, Enumeration,
	Expression, Function, Identifier, Item, ItemData, Model, Output, Pattern, Solve, Type,
	TypeAlias,
};
use crate::{
	file::{ModelRef, SourceFile},
	utils::{arena::ArenaIndex, impl_enum_from, DebugPrint},
};

/// Reference to an item local to a model.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LocalItemRef {
	/// Annotation item ID
	Annotation(ArenaIndex<Item<Annotation>>),
	/// Assignment item ID
	Assignment(ArenaIndex<Item<Assignment>>),
	/// Constraint item ID
	Constraint(ArenaIndex<Item<Constraint>>),
	/// Declaration item ID
	Declaration(ArenaIndex<Item<Declaration>>),
	/// Enumeration item ID
	Enumeration(ArenaIndex<Item<Enumeration>>),
	/// Enum assignment item ID
	EnumAssignment(ArenaIndex<Item<EnumAssignment>>),
	/// Function item ID
	Function(ArenaIndex<Item<Function>>),
	/// Function item ID
	Output(ArenaIndex<Item<Output>>),
	/// Solve item ID
	Solve(ArenaIndex<Item<Solve>>),
	/// Type alias item ID
	TypeAlias(ArenaIndex<Item<TypeAlias>>),
}

impl LocalItemRef {
	/// Get the item data for this item
	pub fn data<'a>(&self, model: &'a Model) -> &'a ItemData {
		match *self {
			LocalItemRef::Annotation(i) => &model[i].data,
			LocalItemRef::Assignment(i) => &model[i].data,
			LocalItemRef::Constraint(i) => &model[i].data,
			LocalItemRef::Declaration(i) => &model[i].data,
			LocalItemRef::Enumeration(i) => &model[i].data,
			LocalItemRef::EnumAssignment(i) => &model[i].data,
			LocalItemRef::Function(i) => &model[i].data,
			LocalItemRef::Output(i) => &model[i].data,
			LocalItemRef::Solve(i) => &model[i].data,
			LocalItemRef::TypeAlias(i) => &model[i].data,
		}
	}
}

impl_enum_from!(LocalItemRef::Annotation(ArenaIndex<Item<Annotation>>));
impl_enum_from!(LocalItemRef::Assignment(ArenaIndex<Item<Assignment>>));
impl_enum_from!(LocalItemRef::Constraint(ArenaIndex<Item<Constraint>>));
impl_enum_from!(LocalItemRef::Declaration(ArenaIndex<Item<Declaration>>));
impl_enum_from!(LocalItemRef::Enumeration(ArenaIndex<Item<Enumeration>>));
impl_enum_from!(LocalItemRef::EnumAssignment(ArenaIndex<Item<EnumAssignment>>));
impl_enum_from!(LocalItemRef::Function(ArenaIndex<Item<Function>>));
impl_enum_from!(LocalItemRef::Output(ArenaIndex<Item<Output>>));
impl_enum_from!(LocalItemRef::Solve(ArenaIndex<Item<Solve>>));
impl_enum_from!(LocalItemRef::TypeAlias(ArenaIndex<Item<TypeAlias>>));

/// Global reference to an item.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ItemRef(salsa::InternId);

impl ItemRef {
	/// Create a new item reference
	pub fn new<T: Into<LocalItemRef>>(db: &dyn Hir, model: ModelRef, item: T) -> Self {
		db.intern_item_ref(ItemRefData(model, item.into()))
	}

	/// Get the model this item is in
	pub fn model_ref(&self, db: &dyn Hir) -> ModelRef {
		db.lookup_intern_item_ref(*self).0
	}

	/// Get the local reference to this item
	pub fn local_item_ref(&self, db: &dyn Hir) -> LocalItemRef {
		db.lookup_intern_item_ref(*self).1
	}

	/// Get the lowered model which contains this item
	pub fn model(&self, db: &dyn Hir) -> Arc<Model> {
		db.lookup_model(self.model_ref(db))
	}
}

impl<'a> DebugPrint<'a> for ItemRef {
	type Database = dyn Hir + 'a;

	fn debug_print(&self, db: &Self::Database) -> String {
		let ItemRefData(model, item) = db.lookup_intern_item_ref(*self);
		let model = db.lookup_model(model);
		match item {
			LocalItemRef::Annotation(i) => model[i].debug_print(db),
			LocalItemRef::Assignment(i) => model[i].debug_print(db),
			LocalItemRef::Constraint(i) => model[i].debug_print(db),
			LocalItemRef::Declaration(i) => model[i].debug_print(db),
			LocalItemRef::Enumeration(i) => model[i].debug_print(db),
			LocalItemRef::EnumAssignment(i) => model[i].debug_print(db),
			LocalItemRef::Function(i) => model[i].debug_print(db),
			LocalItemRef::Output(i) => model[i].debug_print(db),
			LocalItemRef::Solve(i) => model[i].debug_print(db),
			LocalItemRef::TypeAlias(i) => model[i].debug_print(db),
		}
	}
}

impl salsa::InternKey for ItemRef {
	fn from_intern_id(id: salsa::InternId) -> Self {
		Self(id)
	}

	fn as_intern_id(&self) -> salsa::InternId {
		self.0
	}
}

/// Global reference to an item.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ItemRefData(ModelRef, LocalItemRef);

/// Reference to a top-level item of known type.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypedItemRef<T>(ModelRef, ArenaIndex<Item<T>>);

impl<T> TypedItemRef<T> {
	/// Create a new expression reference
	pub fn new(model: ModelRef, item: ArenaIndex<Item<T>>) -> Self {
		Self(model, item)
	}

	/// The model this item is in
	pub fn model_ref(&self) -> ModelRef {
		self.0
	}

	/// The model this item is in
	pub fn model(&self, db: &dyn Hir) -> Arc<Model> {
		db.lookup_model(self.model_ref())
	}

	/// The item local to the model
	pub fn item(&self) -> ArenaIndex<Item<T>> {
		self.1
	}
}

/// Reference to an annotation item
pub type AnnotationRef = TypedItemRef<Annotation>;
/// Reference to an assignment item
pub type AssignmentRef = TypedItemRef<Assignment>;
/// Reference to a constraint item
pub type ConstraintRef = TypedItemRef<Constraint>;
/// Reference to a declaration item
pub type DeclarationRef = TypedItemRef<Declaration>;
/// Reference to an enumeration item
pub type EnumerationRef = TypedItemRef<Enumeration>;
/// Reference to an enumeration item
pub type EnumAssignmentRef = TypedItemRef<EnumAssignment>;
/// Reference to a function item
pub type FunctionRef = TypedItemRef<Function>;
/// Reference to an output item
pub type OutputRef = TypedItemRef<Output>;
/// Reference to a solve item
pub type SolveRef = TypedItemRef<Solve>;
/// Reference to a type alias item
pub type TypeAliasRef = TypedItemRef<TypeAlias>;

impl<T> TypedItemRef<T>
where
	ArenaIndex<Item<T>>: Into<LocalItemRef>,
{
	/// Convert into an `ItemRef`
	pub fn into_item_ref(self, db: &dyn Hir) -> ItemRef {
		ItemRef::new(db, self.0, self.1)
	}
}

/// Global reference to an expression.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ExpressionRef(ItemRef, ArenaIndex<Expression>);

impl ExpressionRef {
	/// Create a new expression reference
	pub fn new(item: ItemRef, e: ArenaIndex<Expression>) -> Self {
		Self(item, e)
	}

	/// Convert into a generic entity reference
	pub fn into_entity(self, db: &dyn Hir) -> EntityRef {
		EntityRef::new(db, self.0, self.1)
	}

	/// Get the item this expression belongs to
	pub fn item(&self) -> ItemRef {
		self.0
	}

	/// Get the index of the expression
	pub fn expression(&self) -> ArenaIndex<Expression> {
		self.1
	}
}

/// Global reference to a type.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypeRef(ItemRef, ArenaIndex<Type>);

impl TypeRef {
	/// Create a new type reference
	pub fn new(item: ItemRef, t: ArenaIndex<Type>) -> Self {
		Self(item, t)
	}

	/// Convert into a generic entity reference
	pub fn into_entity(self, db: &dyn Hir) -> EntityRef {
		EntityRef::new(db, self.0, self.1)
	}

	/// Get the item this type belongs to
	pub fn item(&self) -> ItemRef {
		self.0
	}

	/// Get the index of the type
	pub fn get_type(&self) -> ArenaIndex<Type> {
		self.1
	}
}

/// Global reference to a pattern.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PatternRef(ItemRef, ArenaIndex<Pattern>);

impl PatternRef {
	/// Create a new pattern reference
	pub fn new(item: ItemRef, p: ArenaIndex<Pattern>) -> Self {
		Self(item, p)
	}

	/// Convert into a generic entity reference
	pub fn into_entity(self, db: &dyn Hir) -> EntityRef {
		EntityRef::new(db, self.0, self.1)
	}

	/// Get the item this pattern belongs to
	pub fn item(&self) -> ItemRef {
		self.0
	}

	/// Get the index of the pattern
	pub fn pattern(&self) -> ArenaIndex<Pattern> {
		self.1
	}

	/// Get this pattern as an identifier if it is one
	pub fn identifier(&self, db: &dyn Hir) -> Option<Identifier> {
		let item = self.item();
		let model = item.model(db);
		let data = item.local_item_ref(db).data(&model);
		data[self.pattern()].identifier()
	}
}

/// Local reference to an entity (expression, type, or pattern) owned by an item.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LocalEntityRef {
	/// Expression ID
	Expression(ArenaIndex<Expression>),
	/// Type ID
	Type(ArenaIndex<Type>),
	/// Pattern ID
	Pattern(ArenaIndex<Pattern>),
}

impl_enum_from!(LocalEntityRef::Expression(ArenaIndex<Expression>));
impl_enum_from!(LocalEntityRef::Type(ArenaIndex<Type>));
impl_enum_from!(LocalEntityRef::Pattern(ArenaIndex<Pattern>));

/// Global reference to an entity (expression, type, or pattern)
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct EntityRef(salsa::InternId);

impl EntityRef {
	/// Create a new item reference
	pub fn new<T: Into<LocalEntityRef>>(db: &dyn Hir, item: ItemRef, entity: T) -> Self {
		db.intern_entity_ref(EntityRefData(item, entity.into()))
	}

	/// Get the underlying item refernce
	pub fn item(&self, db: &dyn Hir) -> ItemRef {
		db.lookup_intern_entity_ref(*self).0
	}

	/// Get the local entity reference
	pub fn entity(&self, db: &dyn Hir) -> LocalEntityRef {
		db.lookup_intern_entity_ref(*self).1
	}

	/// Get as an `ExpressionRef` if this is one
	pub fn as_expression_ref(&self, db: &dyn Hir) -> Option<ExpressionRef> {
		match self.entity(db) {
			LocalEntityRef::Expression(e) => Some(ExpressionRef::new(self.item(db), e)),
			_ => None,
		}
	}

	/// Get as an `TypeRef` if this is one
	pub fn as_type_ref(&self, db: &dyn Hir) -> Option<TypeRef> {
		match self.entity(db) {
			LocalEntityRef::Type(t) => Some(TypeRef::new(self.item(db), t)),
			_ => None,
		}
	}

	/// Get as an `PatternRef` if this is one
	pub fn as_pattern_ref(&self, db: &dyn Hir) -> Option<PatternRef> {
		match self.entity(db) {
			LocalEntityRef::Pattern(p) => Some(PatternRef::new(self.item(db), p)),
			_ => None,
		}
	}
}

impl salsa::InternKey for EntityRef {
	fn from_intern_id(id: salsa::InternId) -> Self {
		Self(id)
	}

	fn as_intern_id(&self) -> salsa::InternId {
		self.0
	}
}

/// Global reference to an entity (expression, type, or pattern).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct EntityRefData(ItemRef, LocalEntityRef);

/// Reference to an HIR node (used to map back to AST).
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum NodeRef {
	/// Model reference
	Model(ModelRef),
	/// Item reference
	Item(ItemRef),
	/// Entity reference
	Entity(EntityRef),
}

impl_enum_from!(NodeRef::Model(ModelRef));
impl_enum_from!(NodeRef::Item(ItemRef));
impl_enum_from!(NodeRef::Entity(EntityRef));

impl NodeRef {
	/// Get the inner `ModelRef` if this is one
	pub fn as_model_ref(&self) -> Option<ModelRef> {
		match self {
			NodeRef::Model(m) => Some(*m),
			_ => None,
		}
	}

	/// Get the inner `ItemRef` if this is one
	pub fn as_item_ref(&self) -> Option<ItemRef> {
		match self {
			NodeRef::Item(i) => Some(*i),
			_ => None,
		}
	}

	/// Get the inner `EntityRef` if this is one
	pub fn as_entity(&self) -> Option<EntityRef> {
		match self {
			NodeRef::Entity(e) => Some(*e),
			_ => None,
		}
	}

	/// Get the source and span for emitting a diagnostic
	pub fn source_span(&self, db: &dyn Hir) -> (SourceFile, SourceSpan) {
		let model = match *self {
			NodeRef::Model(m) => m,
			NodeRef::Item(i) => i.model_ref(db),
			NodeRef::Entity(e) => e.item(db).model_ref(db),
		};
		let sm = db.lookup_source_map(model);
		let origin = sm.get_origin(*self).expect("No origin for this node!");
		origin.source_span(db)
	}
}
