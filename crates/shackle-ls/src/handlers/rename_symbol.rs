use std::collections::HashMap;

use lsp_server::{ErrorCode::InvalidRequest, ResponseError};
use lsp_types::{request::Rename, RenameParams, TextEdit, WorkspaceEdit};
use shackle_compiler::{
	db::CompilerDatabase,
	file::ModelRef,
	hir::{
		db::Hir,
		ids::{LocalEntityRef, NodeRef, PatternRef},
		source::{find_node, Point},
	},
	syntax::db::SourceParser,
	utils,
};
use streaming_iterator::StreamingIterator;

use crate::{db::LanguageServerContext, dispatch::RequestHandler, utils::node_ref_to_location};

#[derive(Debug)]
pub struct RenameHandler;

pub struct SymbolHandlerData {
	model_ref: ModelRef,
	cursor_pos: Point,
	new_name: String,
}

fn create_error(msg: &str) -> ResponseError {
	ResponseError {
		code: InvalidRequest as i32,
		message: msg.into(),
		data: None,
	}
}

impl RequestHandler<Rename, SymbolHandlerData> for RenameHandler {
	fn prepare(
		db: &mut impl LanguageServerContext,
		params: RenameParams,
	) -> Result<SymbolHandlerData, ResponseError> {
		// identifiers can consist of anything which is not in these chars
		let cursor_pos = Point {
			row: params.text_document_position.position.line as usize,
			column: params.text_document_position.position.character as usize,
		};

		// cannot include single quotes
		if params.new_name.chars().any(|ch| ch == '\'') {
			return Err(create_error("Identifier cannot include single quotes"));
		}

		// the file it is in
		let model_ref =
			db.set_active_file_from_document(&params.text_document_position.text_document)?;

		// pretty print it to add single quotes, etc as necessary
		let new_name = utils::pretty_print_identifier(&params.new_name);

		Ok(SymbolHandlerData {
			cursor_pos,
			new_name,
			model_ref,
		})
	}

	fn execute(
		db: &CompilerDatabase,
		data: SymbolHandlerData,
	) -> Result<Option<WorkspaceEdit>, ResponseError> {
		// Find the node that is possibly going to be changed
		let node: NodeRef = find_node(db, *data.model_ref, data.cursor_pos, data.cursor_pos)
			.ok_or_else(|| create_error("Identifier not selected"))?;

		let pattern: PatternRef = match node {
			NodeRef::Entity(e) => {
				let item = e.item(db);
				match e.entity(db) {
					LocalEntityRef::Expression(e) => db
						.lookup_item_types(item)
						.name_resolution(e)
						.ok_or_else(|| create_error("Could not resolve pattern"))?,
					LocalEntityRef::Pattern(p) => db
						.lookup_item_types(item)
						.pattern_resolution(p)
						.unwrap_or_else(|| PatternRef::new(item, p)),
					_ => return Ok(None), // Don't want a message in this case, so Ok(None) instead of an Err
				}
			}
			_ => return Ok(None),
		};

		let models = db.resolve_includes().ok().unwrap();
		#[allow(clippy::mutable_key_type)] // Mutable key key required to create WorkspaceEdit
		let mut edits = HashMap::new();

		// loop over all the files included from the main file
		for m in models.iter().copied() {
			let cst = db.cst(*m).ok().unwrap();
			let query = tree_sitter::Query::new(
				&tree_sitter_minizinc::language(),
				tree_sitter_minizinc::IDENTIFIERS_QUERY,
			)
			.expect("Failed to create query");
			let mut cursor = tree_sitter::QueryCursor::new();
			let mut nodes = cursor
				.captures(&query, cst.root_node(), cst.text().as_bytes())
				.map(|(c, _)| c.captures[0].node);
			let source_map = db.lookup_source_map(m);

			// The edits to the current file
			let mut model_edits = Vec::new();
			let mut url = None;

			// Loop over all the identifiers
			while let Some(node) = nodes.next() {
				if let Some(node_ref @ NodeRef::Entity(entity)) = source_map.find_node(*node) {
					let item = entity.item(db);
					let types = db.lookup_item_types(item);
					let def = match entity.entity(db) {
						LocalEntityRef::Expression(e) => types.name_resolution(e),
						LocalEntityRef::Pattern(p) => Some(PatternRef::new(item, p)),
						_ => None,
					};
					// If the definition is matching, push it to be updated
					if def == Some(pattern) {
						if let Some(loc) = node_ref_to_location(db, node_ref) {
							model_edits.push(TextEdit::new(loc.range, data.new_name.clone()));
							url = Some(loc.uri);
						}
					}
				}
			}

			// The file will be known iff there is an edit to change
			if let Some(url) = url {
				// Put it into the hashmap
				edits.insert(url, model_edits);
			}
		}

		Ok(Some(WorkspaceEdit::new(edits)))
	}
}

#[cfg(test)]
mod test {
	use std::str::FromStr;

	use expect_test::expect;
	use lsp_types::{RenameParams, TextDocumentPositionParams, Uri};

	use super::RenameHandler;
	use crate::handlers::test::test_handler;

	#[test]
	fn test_references() {
		test_handler::<RenameHandler, _, _>(
			r#"
var int: x;
any: y = let {
    int: x = 1;
} in x + let {
    int: x = 2;
} in x;
any: z = x;
			"#,
			false,
			RenameParams {
				new_name: "abc 123 !@# \"".into(),
				text_document_position: TextDocumentPositionParams {
					text_document: lsp_types::TextDocumentIdentifier {
						uri: Uri::from_str("file:///test.mzn").unwrap(),
					},
					position: lsp_types::Position {
						line: 6,
						character: 6,
					},
				},
				work_done_progress_params: lsp_types::WorkDoneProgressParams {
					work_done_token: None,
				},
			},
			expect!([r#"
    {
      "Ok": {
        "changes": {
          "test.mzn": [
            {
              "range": {
                "start": {
                  "line": 5,
                  "character": 9
                },
                "end": {
                  "line": 5,
                  "character": 10
                }
              },
              "newText": "'abc 123 !@# \"'"
            },
            {
              "range": {
                "start": {
                  "line": 6,
                  "character": 5
                },
                "end": {
                  "line": 6,
                  "character": 6
                }
              },
              "newText": "'abc 123 !@# \"'"
            }
          ]
        }
      }
    }"#]),
		)
	}
}
