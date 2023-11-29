from app import RAGHandler


def main():
    # rag_handler = RAGHandler()
    # rag_handler = RAGHandler(chunk_size=400, chunk_overlap=100, temperature=0.2, max_tokens=200)
    rag_handler = RAGHandler(chunk_size=200, chunk_overlap=100, temperature=0.2, max_tokens=200)
    query = ""
    while True:
        user_input = input("Ask something about the document (or 'exit' or 'debug'): ")

        if user_input.lower() == "debug":
            if query:
                rag_handler._debug(query)
            continue
        if user_input.lower() == "exit":
            print("Exiting...")
            break  # Exit script

        query = user_input
        print(rag_handler.ask(query))


if __name__ == "__main__":
    main()
