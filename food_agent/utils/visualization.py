from langgraph.graph.state import CompiledStateGraph


def draw_graph(agent: CompiledStateGraph):
    try:
        with open("graph.png", "wb") as f:
            f.write(agent.get_graph().draw_mermaid_png())
            print("Saving")
    except Exception as e:
        print(f"Saving graph visualization failed: {e}")
