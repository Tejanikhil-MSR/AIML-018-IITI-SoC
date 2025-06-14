import pathway as pw

def setup_webserver(host="0.0.0.0", port=8011):
    return pw.io.http.PathwayWebserver(host=host, port=port)

class QuerySchema(pw.Schema):
    messages: str

def setup_rest_connector(webserver):
    return pw.io.http.rest_connector(
        webserver=webserver,
        schema=QuerySchema,
        autocommit_duration_ms=50,
        delete_completed_queries=False,
    )
