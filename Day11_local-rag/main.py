from config_loader import load_config
from ingestion_retrieval import ingest, retrieve, VectorStore
from generation import generate_answer

def main():
    cfg = load_config()

    # Create ONE shared in-memory vector store
    vs = VectorStore(cfg)

    while True:
        print("\n1) Ingest documents")
        print("2) Ask questions")
        print("3) Exit")

        choice = input("Select option (1/2/3): ").strip()

        if choice == "1":
            print("\n--- Ingesting documents ---\n")
            ingest(cfg, vs)
            print("\nIngestion complete.\n")

        elif choice == "2":
            print("\n--- Question Answering ---\n")
            while True:
                q = input("Question (or 'back'): ").strip()
                if q.lower() == "back":
                    break

                context_chunks = retrieve(q, cfg, vs)
                answer = generate_answer(q, context_chunks, cfg)

                print("\nAnswer:\n")
                print(answer)
                print("\n---------------------------\n")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
