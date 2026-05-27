import os
import json

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False


class RAGEngine:
    def __init__(self, db_path='data/chroma_db', model_name='all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.model_name = model_name
        self.collection = None
        self.embedding_model = None
        self.initialized = False

        if not HAS_RAG_DEPS:
            print("WARNING: chromadb or sentence-transformers not installed.")
            print("RAG functionality will be disabled. Install with:")
            print("pip install chromadb sentence-transformers")
            return

        try:
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="materials_science",
                metadata={"hnsw:space": "cosine"}
            )
            self.initialized = True
            print(f"Engine initialized. Collection has {self.collection.count()} documents.")
        except Exception as e:
            print(f"Initialization failed: {e}")

    def populate_from_json(self, json_path):
        if not self.initialized:
            print("Engine not initialized. Skipping.")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)

        if self.collection.count() >= len(docs):
            print(f"Knowledge base already populated ({self.collection.count()} docs). Skipping.")
            return

        texts = [d['text'] for d in docs]
        ids = [d['id'] for d in docs]
        metadatas = [d.get('metadata', {}) for d in docs]
        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.upsert(documents=texts, embeddings=embeddings,
                               ids=ids, metadatas=metadatas)
        print(f"Populated {len(docs)} documents into vector store.")

    def query(self, question, n_results=3):
        if not self.initialized:
            return self._fallback_context(question)

        query_embedding = self.embedding_model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        contexts = []
        if results and results['documents']:
            for doc in results['documents'][0]:
                contexts.append(doc)

        return '\n\n'.join(contexts) if contexts else self._fallback_context(question)

    def _fallback_context(self, question):
        return (
            "The Al-Zn binary system exhibits several key phases: FCC_A1 (aluminum-rich solid solution), "
            "HCP_ZN (zinc-rich solid solution), and LIQUID. The system features a eutectic reaction near "
            "X(Zn)=0.95 at approximately 655K. At high temperatures, a single liquid phase is stable. "
            "The miscibility gap in the FCC phase leads to a monotectoid reaction. Phase stability is "
            "governed by Gibbs free energy minimization, where G = H - TS. At higher temperatures, "
            "the entropy term (TS) dominates, favoring disordered phases."
        )


_engine_instance = None

def get_rag_engine(db_path='data/chroma_db'):
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(db_path)
    return _engine_instance
