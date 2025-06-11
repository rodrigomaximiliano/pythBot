from typing import Dict, Any, List

_chat_histories: Dict[str, List[Any]] = {}

def get_history(session_id: str) -> List[Any]:
    return _chat_histories.get(session_id, [])

def append_history(session_id: str, chat_history_ids: Any):
    if session_id in _chat_histories:
        _chat_histories[session_id].append(chat_history_ids)
    else:
        _chat_histories[session_id] = [chat_history_ids]
