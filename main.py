"""主启动器：仅负责热键注册和应用启动。"""

import keyboard

from workflows import VoiceChatWorkflow


def main():
    workflow = VoiceChatWorkflow()

    print("[system] Assistant ready.")
    print("Hotkeys:")
    print("  Alt+1 -> 开始/结束普通录音")
    print("  Alt+2 -> 开始/结束截图录音")
    print("  Alt+3 -> 强制停止当前录音(不发送)")
    print("  Alt+0 -> 清空对话与截图")
    print("  Esc   -> 退出")

    keyboard.add_hotkey("alt+1", lambda: workflow.start_or_stop_recording(with_screenshot=False))
    keyboard.add_hotkey("alt+2", lambda: workflow.start_or_stop_recording(with_screenshot=True))
    keyboard.add_hotkey("alt+3", workflow.force_stop_only)
    keyboard.add_hotkey("alt+0", workflow.clear_history)

    keyboard.wait("esc")
    print("[system] Exiting.")


if __name__ == "__main__":
    main()
