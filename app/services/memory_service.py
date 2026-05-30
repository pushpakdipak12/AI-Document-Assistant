from collections import defaultdict


conversation_memory = defaultdict(list)


def save_message(session_id: str, role: str, content: str):
    conversation_memory[session_id].append(
        {
            "role": role,
            "content": content
        }
    )


def get_conversation_history(session_id: str):
    return conversation_memory[session_id]


def clear_session(session_id: str):
    conversation_memory[session_id] = []