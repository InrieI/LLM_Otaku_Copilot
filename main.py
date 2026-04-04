"""
LLM_OtakuWifu_Copilot 主程序
一个 AI 桌面助手，可以通过快捷键唤醒，回答你的问题

功能：
1. 按 Ctrl+Shift+A (可改) - 唤醒助手
2. 列出本地可用的 AI 模型
3. 接收用户输入或语音
4. 调用 AI 生成回答
5. 显示结果
"""

import keyboard
import time


def on_trigger():
    """
    当快捷键被按下时调用这个函数
    这里将来会放入：
    - 录音逻辑
    - 语音转文字
    - AI 回答逻辑
    - 语音合成等
    """
    print("\n[系统] 检测到快捷键！准备开始录音和截图...")
    # 这里以后会放录音和截图的代码
    print("[系统] 正在处理中... (模拟)")
    time.sleep(1)
    print("[系统] 处理完成，等待下一次指令。")


# 启动提示
print("助手已启动！")
print("按下 'Ctrl + Shift + A' 触发功能 (按 'Esc' 退出)")

# 绑定快捷键：按 Ctrl+Shift+A 时调用 on_trigger() 函数
keyboard.add_hotkey('ctrl+shift+a', on_trigger)

# keyboard.wait('esc') 会让程序一直运行，直到你按下 Esc 键
# 这样程序就能不断监听快捷键
keyboard.wait('esc')

print("[系统] 程序退出")
