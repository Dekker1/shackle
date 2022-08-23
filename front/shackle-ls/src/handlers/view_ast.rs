use lsp_server::ResponseError;
use lsp_types::TextDocumentPositionParams;
use shackle::{db::CompilerDatabase, file::ModelRef, syntax::db::SourceParser};

use crate::{dispatch::RequestHandler, extensions::ViewAst, LanguageServerDatabase};

#[derive(Debug)]
pub struct ViewAstHandler;

impl RequestHandler<ViewAst, ModelRef> for ViewAstHandler {
	fn prepare(
		db: &mut LanguageServerDatabase,
		params: TextDocumentPositionParams,
	) -> Result<ModelRef, ResponseError> {
		db.set_active_file_from_document(&params.text_document)
	}
	fn execute(db: &CompilerDatabase, model_ref: ModelRef) -> Result<String, ResponseError> {
		match db.ast(*model_ref) {
			Ok(ast) => Ok(format!("{:#?}", ast)),
			Err(e) => Ok(e.to_string()),
		}
	}
}
