//! Destructuring/case matching patterns
//!

use super::{db::Hir, BooleanLiteral, FloatLiteral, IntegerLiteral, ItemData, StringLiteral};
use crate::db::{InternedString, InternedStringData};
use crate::{arena::ArenaIndex, utils::impl_enum_from};

/// A pattern for destructuring
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Pattern {
	/// A single identifier
	Identifier(Identifier),
	/// Don't care wildcard
	Anonymous,
	/// Absent literal
	Absent,
	/// Boolean literal
	Boolean(BooleanLiteral),
	/// Float literal
	Float {
		/// Whether this has been negated
		negated: bool,
		/// The literal value
		value: FloatLiteral,
	},
	/// Integer literal
	Integer {
		/// Whether this has been negated
		negated: bool,
		/// The literal value
		value: IntegerLiteral,
	},
	/// Infinity
	Infinity {
		/// Whether this has been negated
		negated: bool,
	},
	/// String literal
	String(StringLiteral),
	/// Enum constructor pattern
	Call {
		/// Callee identifier
		function: ArenaIndex<Pattern>,
		/// Call arguments
		arguments: Box<[ArenaIndex<Pattern>]>,
	},
	/// Tuple pattern
	Tuple {
		/// Tuple fields
		fields: Box<[ArenaIndex<Pattern>]>,
	},
	/// Record pattern
	Record {
		/// Record fields (pairs of field name, field value pattern)
		fields: Box<[(Identifier, ArenaIndex<Pattern>)]>,
	},
	/// Indicates an error
	Missing,
}

impl_enum_from!(Pattern::Identifier);
impl_enum_from!(Pattern::Boolean(BooleanLiteral));
impl_enum_from!(Pattern::String(StringLiteral));

impl Pattern {
	/// Get the identifier if this is one
	pub fn identifier(&self) -> Option<Identifier> {
		match *self {
			Pattern::Identifier(i) => Some(i),
			_ => None,
		}
	}

	/// Get the identifiers in this pattern
	pub fn identifiers(
		pattern: ArenaIndex<Pattern>,
		data: &ItemData,
	) -> impl '_ + Iterator<Item = ArenaIndex<Pattern>> {
		let mut todo = vec![pattern];
		std::iter::from_fn(move || {
			while let Some(p) = todo.pop() {
				match &data[p] {
					Pattern::Identifier(_) => return Some(p),
					Pattern::Call { arguments, .. } => todo.extend(arguments.iter().copied()),
					Pattern::Tuple { fields } => todo.extend(fields.iter().copied()),
					Pattern::Record { fields } => todo.extend(fields.iter().map(|(_, p)| *p)),
					_ => (),
				}
			}
			None
		})
	}

	/// Get whether this pattern can only possibly match a single value
	/// (i.e. no identifiers, no wildcards)
	pub fn is_singular(pattern: ArenaIndex<Pattern>, data: &ItemData) -> bool {
		let mut todo = vec![pattern];
		while let Some(p) = todo.pop() {
			match &data[p] {
				Pattern::Identifier(_) | Pattern::Anonymous => return false,
				Pattern::Call { arguments, .. } => todo.extend(arguments.iter().copied()),
				Pattern::Tuple { fields } => todo.extend(fields.iter().copied()),
				Pattern::Record { fields } => todo.extend(fields.iter().map(|(_, p)| *p)),
				_ => (),
			}
		}
		true
	}
}

/// Identifier
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier(pub InternedString);

impl Identifier {
	/// Create a new identifier with the given value
	pub fn new<T: Into<InternedStringData>>(v: T, db: &dyn Hir) -> Self {
		Self(db.intern_string(v.into()))
	}

	/// Get the name of this identifier
	pub fn lookup(&self, db: &dyn Hir) -> String {
		db.lookup_intern_string(self.0).0
	}

	/// Append ⁻¹ to this identifier
	pub fn inversed(&self, db: &dyn Hir) -> Self {
		let mut v = self.lookup(db);
		v.push_str("⁻¹");
		Self::new(v, db)
	}

	/// Whether this identifier matches a string
	pub fn is<T: Into<InternedStringData>>(&self, db: &dyn Hir, v: T) -> bool {
		db.intern_string(v.into()) == self.0
	}

	/// Pretty print this identifier (adding quotes if needed)
	///
	/// TODO: Don't quote UTF-8
	pub fn pretty_print(&self, db: &dyn Hir) -> String {
		let ident = self.lookup(db);
		let name = ident.as_str();
		if matches!(
			name,
			"ann"
				| "annotation" | "any"
				| "array" | "bool"
				| "case" | "constraint"
				| "default" | "diff"
				| "div" | "else" | "elseif"
				| "endif" | "enum"
				| "false" | "float"
				| "function" | "if"
				| "in" | "include"
				| "int" | "intersect"
				| "let" | "list" | "maximize"
				| "minimize" | "mod"
				| "not" | "of" | "op"
				| "opt" | "output"
				| "par" | "predicate"
				| "record" | "satisfy"
				| "set" | "solve"
				| "string" | "subset"
				| "superset" | "symdiff"
				| "test" | "then"
				| "true" | "tuple"
				| "type" | "union"
				| "var" | "where"
				| "xor"
		) {
			return format!("'{}'", name);
		}
		for c in name.chars() {
			if matches!(
				c,
				'"' | '\''
					| '.' | '-' | '[' | ']'
					| '^' | ',' | ';' | ':'
					| '(' | ')' | '{' | '}'
					| '&' | '|' | '$' | '∞'
					| '%' | '<' | '>' | '⟷'
					| '⇔' | '→' | '⇒' | '←'
					| '⇐' | '/' | '∨' | '⊻'
					| '∧' | '=' | '!' | '≠'
					| '≤' | '≥' | '∈' | '⊆'
					| '⊇' | '∪' | '∩' | '+'
					| '*' | '~'
			) || c.is_whitespace()
			{
				return format!("'{}'", name);
			}
		}
		ident
	}
}

impl From<Identifier> for InternedString {
	fn from(ident: Identifier) -> Self {
		ident.0
	}
}
