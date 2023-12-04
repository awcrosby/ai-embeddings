from app import RAGHandler

import time


def main():
    start_time = time.time()
    # rag_handler = RAGHandler()
    # rag_handler = RAGHandler(chunk_size=400, chunk_overlap=100, temperature=0.2, max_tokens=200)
    rag_handler = RAGHandler(chunk_size=400, chunk_overlap=100, temperature=0.2, max_tokens=200)
    print(f"RAGHandler initialized in {time.time() - start_time} seconds")
    while True:
        user_input = input("Ask something about the document (or 'exit' or 'debug'): ")

        match user_input.lower():
            case "debug":
                rag_handler._debug()
            case "exit":
                print("Exiting...")
                break
            case _:
                print(rag_handler.ask(user_input)["answer"])


if __name__ == "__main__":
    main()
