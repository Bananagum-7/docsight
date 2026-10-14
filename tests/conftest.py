import pytest

from docsight.models import Chunk


def make_chunk(i: int, crumb: str, body: str, path: str = "tutorial/x.md") -> Chunk:
    return Chunk(
        id=f"c{i}",
        text=f"[{crumb}]\n{body}",
        doc_path=path,
        title=crumb.split(" > ")[0],
        breadcrumb=crumb,
        url=f"https://example.com/{path[:-3]}/",
        has_code="```" in body,
    )


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        make_chunk(1, "Background Tasks > Using BackgroundTasks", "Import BackgroundTasks and call add_task to run work after returning a response.", "tutorial/background-tasks.md"),
        make_chunk(2, "CORS > Use CORSMiddleware", "Add CORSMiddleware with allow_origins to let browsers call your API from another origin.", "tutorial/cors.md"),
        make_chunk(3, "Dependencies with yield", "Use yield in a dependency to run cleanup code, such as closing a database session, after the response.", "tutorial/dependencies/dependencies-with-yield.md"),
        make_chunk(4, "Testing > Using TestClient", "Create a TestClient with your app and call client.get to test endpoints with pytest.", "tutorial/testing.md"),
        make_chunk(5, "Request Files > UploadFile", "Declare a parameter of type UploadFile to receive uploaded files; it uses a spooled file.", "tutorial/request-files.md"),
        make_chunk(6, "Security > OAuth2 with JWT", "Hash passwords with pwdlib and create JWT access tokens signed with a secret key.", "tutorial/security/oauth2-jwt.md"),
    ]
