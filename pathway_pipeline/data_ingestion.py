import pathway as pw

def read_documents(data_path):
    return pw.io.fs.read(data_path, format="binary", with_metadata=True)