# Stub de compatibilidade - conexões S3/OCI desativadas no deploy público
def get_duckdb_connection():
    raise NotImplementedError("Conexão com S3 não disponível neste ambiente.")

def get_s3_client():
    raise NotImplementedError("Conexão com S3 não disponível neste ambiente.")