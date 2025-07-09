import pathway as pw
import logging

# Basic configuration
logging.basicConfig(filename='pathway_debug.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )

logging.debug("This is a debug message")     # Not shown by default

class StudentSchema(pw.Schema):
    id: int
    name: str
    score: float

# Read the json using pathway and convert into pathway table
students = pw.io.jsonlines.read("Students.json", schema=StudentSchema, mode="static", with_metadata=True)

# Start a webserver that accepts the post requests from the client
web_server = pw.io.http.PathwayWebserver(host="127.0.0.1", port=9999)
# This will return the user input in a table whoes schema is defined in schema as well as a sink
words, response_writer = pw.io.http.rest_connector(webserver=web_server, route="/uppercase", schema=StudentSchema)

# Preprocessing of the user input happens here (in terms of table only)
uppercase_words = words.select(query_id=pw.this.id, result=pw.apply(lambda x: x.upper(), pw.this.name))

# response writer will take of returning the output to the user
response_writer(uppercase_words)

def change(key: pw.Pointer, row: dict, time: int, is_addition: bool):
  logging.info(f"{key}, {row}, {time}, {is_addition}")
  
def on_end():
  logging.info("End of stream.")
  
pw.io.subscribe(students, on_change=change) # Function on_change will be called on every change in the input stream.
# pw.io.subscribe(students, on_change, on_end) # on_end function is triggered only once

pw.run() # Remember that pathway is a lazy evaluator which means we need to trigger an action which is pw.run()