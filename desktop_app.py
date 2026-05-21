from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import customtkinter as ctk
import requests


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BASE_URL = "http://127.0.0.1:8000"
NEW_CHAT_TITLE = "新对话"


@dataclass
class ChatSession:
    session_id: str
    title: str
    chat_history: list[dict[str, str]] = field(default_factory=list)
    clinical_summary: str = ""
    messages: list[tuple[str, str]] = field(default_factory=list)


class ChatApp(ctk.CTk):
    """DeepOnco AI desktop client with stable streaming text output."""

    def __init__(self) -> None:
        super().__init__()
        self.title("DeepOnco AI - 智能肿瘤医学大脑")
        self.geometry("1180x720")
        self.minsize(1060, 640)
        self.configure(fg_color="#0d1521")
        self.base_url = BASE_URL

        self.sessions: dict[str, ChatSession] = {}
        self.current_session_id = ""
        self.session_buttons: dict[str, ctk.CTkButton] = {}
        self.streaming_session_id: str | None = None
        self.streaming_message_index: int | None = None
        self.streaming_display_lengths: dict[tuple[str, int], int] = {}
        self._stream_update_lock = threading.Lock()
        self._stream_update_pending = False

        self._set_window_icon()
        self.grid_columnconfigure(0, weight=0, minsize=310)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_chat_area()
        self.new_chat()
        self._refresh_service_status()

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, corner_radius=12, fg_color="#172231")
        self.sidebar.grid(row=0, column=0, padx=(18, 10), pady=18, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)

        logo = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#1c2a3a")
        logo.grid(row=0, column=0, padx=16, pady=(16, 12), sticky="ew")
        ctk.CTkLabel(
            logo,
            text="DeepOnco AI",
            font=("Microsoft YaHei UI", 26, "bold"),
            text_color="#55d7e9",
            anchor="w",
        ).pack(padx=18, pady=(18, 2), anchor="w")
        ctk.CTkLabel(
            logo,
            text="肿瘤科诊疗辅助系统",
            font=("Microsoft YaHei UI", 14),
            text_color="#d6e4ef",
            anchor="w",
        ).pack(padx=18, pady=(0, 18), anchor="w")

        self.service_label = ctk.CTkLabel(
            self.sidebar,
            text="正在检测服务...",
            font=("Microsoft YaHei UI", 13),
            text_color="#f2c94c",
            anchor="w",
        )
        self.service_label.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="ew")

        self.new_chat_button = ctk.CTkButton(
            self.sidebar,
            text="+ 新建对话",
            height=40,
            font=("Microsoft YaHei UI", 15, "bold"),
            command=self.new_chat,
        )
        self.new_chat_button.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="ew")

        self.session_list = ctk.CTkScrollableFrame(self.sidebar, fg_color="#101b2a", corner_radius=10)
        self.session_list.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.session_list.grid_columnconfigure(0, weight=1)

    def _build_chat_area(self) -> None:
        self.chat_panel = ctk.CTkFrame(self, corner_radius=12, fg_color="#162334")
        self.chat_panel.grid(row=0, column=1, padx=(6, 18), pady=18, sticky="nsew")
        self.chat_panel.grid_columnconfigure(0, weight=1)
        self.chat_panel.grid_rowconfigure(1, weight=1)

        self.chat_title = ctk.CTkLabel(
            self.chat_panel,
            text=NEW_CHAT_TITLE,
            font=("Microsoft YaHei UI", 20, "bold"),
            text_color="#f3f8ff",
            anchor="w",
        )
        self.chat_title.grid(row=0, column=0, padx=18, pady=(14, 10), sticky="ew")

        self.chat_box = ctk.CTkTextbox(
            self.chat_panel,
            corner_radius=0,
            fg_color="#101b2a",
            font=("Microsoft YaHei UI", 14),
            wrap="word",
            state="disabled",
        )
        self.chat_box.grid(row=1, column=0, sticky="nsew")
        self._configure_chat_tags()

        bottom = ctk.CTkFrame(self.chat_panel, corner_radius=0, fg_color="#162334")
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(
            bottom,
            placeholder_text="描述症状、药物、检查或饮食问题...",
            font=("Microsoft YaHei UI", 14),
            height=46,
            fg_color="#0f1a29",
            border_color="#3b5068",
        )
        self.input_entry.grid(row=0, column=0, padx=(18, 10), pady=(16, 8), sticky="ew")
        self.input_entry.bind("<Return>", lambda _: self.send_message())

        self.send_button = ctk.CTkButton(
            bottom,
            text="发送",
            width=110,
            height=46,
            font=("Microsoft YaHei UI", 15, "bold"),
            command=self.send_message,
        )
        self.send_button.grid(row=0, column=1, padx=(0, 18), pady=(16, 8), sticky="e")

        self.status_label = ctk.CTkLabel(
            bottom,
            text="就绪",
            font=("Microsoft YaHei UI", 12),
            text_color="#63d989",
            anchor="w",
        )
        self.status_label.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="ew")

    def new_chat(self) -> None:
        session_id = uuid.uuid4().hex
        session = ChatSession(
            session_id=session_id,
            title=NEW_CHAT_TITLE,
            messages=[("assistant", "您好，我是 DeepOnco AI。请直接输入您的问题。")],
        )
        self.sessions[session_id] = session
        self.current_session_id = session_id
        self._render_session_buttons()
        self._render_current_session()
        self.input_entry.focus_set()

    def switch_session(self, session_id: str) -> None:
        if session_id not in self.sessions or session_id == self.current_session_id:
            return
        self.current_session_id = session_id
        self.streaming_session_id = None
        self.streaming_message_index = None
        self._render_session_buttons()
        self._render_current_session()

    def send_message(self) -> None:
        query = self.input_entry.get().strip()
        if not query:
            return

        session = self._current_session()
        if session.title == NEW_CHAT_TITLE:
            session.title = query[:18] + ("..." if len(query) > 18 else "")
            self._render_session_buttons()

        session.messages.append(("user", query))
        self.input_entry.delete(0, "end")
        self._render_current_session()
        self._set_requesting_state(True)
        threading.Thread(target=self._request_backend, args=(session.session_id, query), daemon=True).start()

    def _request_backend(self, session_id: str, query: str) -> None:
        session = self.sessions[session_id]
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/chat/stream",
                json={
                    "query": query,
                    "chat_history": session.chat_history,
                    "clinical_summary": session.clinical_summary,
                },
                stream=True,
                timeout=(5, 240),
            )
            response.raise_for_status()

            assistant_index: int | None = None
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("type")
                if event_type == "status":
                    message = str(event.get("message", "正在处理..."))
                    self.after(0, lambda text=message: self.status_label.configure(text=text))
                elif event_type == "start":
                    assistant_index = len(session.messages)
                    session.messages.append(("assistant", ""))
                    self.after(0, self._start_stream_message, session_id, assistant_index)
                elif event_type == "token":
                    token = str(event.get("content", ""))
                    if assistant_index is None:
                        assistant_index = len(session.messages)
                        session.messages.append(("assistant", ""))
                        self.after(0, self._start_stream_message, session_id, assistant_index)
                    role, old_text = session.messages[assistant_index]
                    session.messages[assistant_index] = (role, old_text + token)
                    self._schedule_stream_update(session_id, assistant_index)
                elif event_type == "done":
                    session.chat_history = event.get("chat_history", session.chat_history)
                    session.clinical_summary = event.get("clinical_summary", session.clinical_summary)
                    self.after(0, self._finish_stream, session_id)
            status = "回答完成"
        except requests.exceptions.RequestException as exc:
            status = "请求失败"
            session.messages.append(("assistant", f"连接 AI 服务失败：{exc}"))
            self.after(0, self._render_current_session)
        except json.JSONDecodeError as exc:
            status = "响应解析失败"
            session.messages.append(("assistant", f"后端返回内容不是合法 JSON：{exc}"))
            self.after(0, self._render_current_session)
        finally:
            self.after(0, lambda: self.status_label.configure(text=status))
            self.after(0, self._set_requesting_state, False)

    def _start_stream_message(self, session_id: str, message_index: int) -> None:
        if session_id != self.current_session_id:
            return
        self.streaming_session_id = session_id
        self.streaming_message_index = message_index
        self.streaming_display_lengths[(session_id, message_index)] = 0
        self._append_chat_text("\nDeepOnco AI\n", "assistant_name")
        self._scroll_to_bottom()

    def _schedule_stream_update(self, session_id: str, message_index: int) -> None:
        with self._stream_update_lock:
            if self._stream_update_pending:
                return
            self._stream_update_pending = True
        self.after(60, self._flush_stream_update, session_id, message_index)

    def _flush_stream_update(self, session_id: str, message_index: int) -> None:
        with self._stream_update_lock:
            self._stream_update_pending = False
        self._append_pending_stream_text(session_id, message_index)

    def _append_pending_stream_text(self, session_id: str, message_index: int) -> None:
        if session_id != self.current_session_id:
            return
        session = self.sessions[session_id]
        text = session.messages[message_index][1]
        key = (session_id, message_index)
        displayed_length = self.streaming_display_lengths.get(key, 0)
        delta = text[displayed_length:]
        if not delta:
            return
        self._append_chat_text(delta, "assistant")
        self.streaming_display_lengths[key] = len(text)
        self._scroll_to_bottom()

    def _finish_stream(self, session_id: str) -> None:
        if session_id == self.current_session_id and self.streaming_message_index is not None:
            self._append_pending_stream_text(session_id, self.streaming_message_index)
            self._append_chat_text("\n", "assistant")
            self._scroll_to_bottom()
        self.streaming_session_id = None
        self.streaming_message_index = None
        with self._stream_update_lock:
            self._stream_update_pending = False

    def _render_current_session(self) -> None:
        session = self._current_session()
        self.chat_title.configure(text=session.title)
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        for index, (role, text) in enumerate(session.messages):
            if role == "user":
                self.chat_box.insert("end", "患者\n", "user_name")
                self.chat_box.insert("end", f"{text}\n\n", "user")
            else:
                self.chat_box.insert("end", "DeepOnco AI\n", "assistant_name")
                self.chat_box.insert("end", f"{text}\n\n", "assistant")
                if session.session_id == self.streaming_session_id and index == self.streaming_message_index:
                    self.streaming_display_lengths[(session.session_id, index)] = len(text)
        self.chat_box.configure(state="disabled")
        self._scroll_to_bottom()

    def _render_session_buttons(self) -> None:
        for child in self.session_list.winfo_children():
            child.destroy()
        self.session_buttons = {}
        for row, session in enumerate(self.sessions.values()):
            is_current = session.session_id == self.current_session_id
            button = ctk.CTkButton(
                self.session_list,
                text=session.title,
                height=36,
                anchor="w",
                fg_color="#2d6cdf" if is_current else "#1b2a3b",
                hover_color="#31516e",
                command=lambda sid=session.session_id: self.switch_session(sid),
            )
            button.grid(row=row, column=0, padx=8, pady=(8 if row == 0 else 4, 4), sticky="ew")
            self.session_buttons[session.session_id] = button

    def _append_chat_text(self, text: str, tag: str) -> None:
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text, tag)
        self.chat_box.configure(state="disabled")

    def _scroll_to_bottom(self) -> None:
        self.chat_box.see("end")

    def _configure_chat_tags(self) -> None:
        try:
            self.chat_box.tag_config("user_name", foreground="#93c5fd", spacing1=12)
            self.chat_box.tag_config("assistant_name", foreground="#67e8f9", spacing1=12)
            self.chat_box.tag_config("user", foreground="#f4f7fb", lmargin1=18, lmargin2=18, rmargin=18, spacing3=8)
            self.chat_box.tag_config("assistant", foreground="#f5fbff", lmargin1=18, lmargin2=18, rmargin=18, spacing3=8)
        except Exception:
            pass

    def _current_session(self) -> ChatSession:
        return self.sessions[self.current_session_id]

    def _set_requesting_state(self, is_requesting: bool) -> None:
        state = "disabled" if is_requesting else "normal"
        self.send_button.configure(state=state, text="生成中" if is_requesting else "发送")
        self.input_entry.configure(state=state)
        if is_requesting:
            self.status_label.configure(text="正在检索知识库并生成回答...")

    def _refresh_service_status(self) -> None:
        threading.Thread(target=self._health_worker, daemon=True).start()

    def _health_worker(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            response.raise_for_status()
            self.after(0, lambda: self.service_label.configure(text="已连接服务", text_color="#63d989"))
        except requests.exceptions.RequestException:
            self.after(0, lambda: self.service_label.configure(text="未连接，请确认后端服务已启动", text_color="#ff6b7a"))

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).with_name("icon.ico")
        if icon_path.exists():
            self.iconbitmap(str(icon_path))


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
