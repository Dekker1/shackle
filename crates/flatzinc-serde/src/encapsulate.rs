//! Helper structures to encapsulate certain types in the FlatZinc JSON
//! serialization

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::RangeList;

/// Encapsulated String helper struct
#[derive(Deserialize, Serialize)]
#[serde(rename = "string")]
struct StringLiteral {
	/// content of the string literal
	string: String,
}
/// Deserialization function to resolve the encapsulation of string literals in
/// the FlatZinc serialization format
pub(crate) fn deserialize_encapsulated_string<'de, D: Deserializer<'de>>(
	deserializer: D,
) -> Result<String, D::Error> {
	let s: StringLiteral = Deserialize::deserialize(deserializer)?;
	Ok(s.string)
}

/// Serialization function to be used for the encapsulation of string literals
/// required by the FlatZinc serialization format
pub(crate) fn serialize_encapsulate_string<S: Serializer>(
	s: &str,
	serializer: S,
) -> Result<S::Ok, S::Error> {
	Serialize::serialize(
		&StringLiteral {
			string: String::from(s),
		},
		serializer,
	)
}

/// Encapsulated set helper struct
#[derive(Deserialize, Serialize)]
#[serde(rename = "set")]
struct SetLiteral<E: PartialOrd> {
	/// RangeList used to represent the content of the set
	set: RangeList<E>,
}
/// Deserialization function to resolve the encapsulation of set literals in the
/// FlatZinc serialization format
pub(crate) fn deserialize_encapsulated_set<
	'de,
	D: Deserializer<'de>,
	E: PartialOrd + Deserialize<'de>,
>(
	deserializer: D,
) -> Result<RangeList<E>, D::Error> {
	let s: SetLiteral<E> = Deserialize::deserialize(deserializer)?;
	Ok(s.set)
}

/// Serialization function to be used for the encapsulation of set literals
/// required by the FlatZinc serialization format
pub(crate) fn serialize_encapsulate_set<E: PartialOrd + Serialize + Clone, S: Serializer>(
	r: &RangeList<E>,
	serializer: S,
) -> Result<S::Ok, S::Error> {
	Serialize::serialize(&SetLiteral { set: r.clone() }, serializer)
}
