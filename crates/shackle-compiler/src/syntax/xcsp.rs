//! XCSP AST representation
//!
//! Respresentation of a deserialized XCSP3 file. This module mainly introduces
//! type aliases for structures in xcsp3-serde with all generics instantiated as
//! to be used within shackle.

use xcsp3_serde::{
	constraint::Constraint,
	expression::{BoolExp, Exp, IntExp, SetExp},
	Array, Instance, Objective, Objectives, Variable,
};

/// XCSP3 Array Representation
pub type XcspArray = Array<String>;
/// XCSP3 Boolean Expression Representation
pub type XcspBoolExp = BoolExp<String>;
/// XCSP3 Constraint Representation
pub type XcspConstraint = Constraint<String>;
/// XCSP3 Untyped Expression Representation
pub type XcspExp = Exp<String>;
/// XCSP3 Integer Expression Representation
pub type XcspIntExp = IntExp<String>;
/// XCSP3 Model Representation
pub type XcspModel = Instance<String>;
/// XCSP3 Objective Representation
pub type XcspObjective = Objective<String>;
/// XCSP3 Container for Multiple Objectives (and their combining goal)
pub type XcspObjectives = Objectives<String>;
/// XCSP3 Set Expression Representation
pub type XcspSetExp = SetExp<String>;
/// XCSP3 Variable Reference Representation
pub type XcspVariable = Variable<String>;
