# -*- coding: utf-8 -*-
import socket
import base64
import re
import sys
import unicodedata
import json
from typing import List, Dict, Optional, Tuple
import time

"""
云端任务客户端：封装向云端发送任务的完整流程，并处理所有返回信息。（Python 等价实现）
保持与原 Java 版本“完全相同”的逻辑分支与返回语句。
"""

# ======== 云端地址和端口（需根据实际部署修改）========
CLOUD_IP = "47.115.134.188"   # 【阿里云】
CLOUD_PORT = 13344            # 【阿里云】
    

def _remove_format_chars(s: str) -> str:
    """
    去掉 Unicode 中类别为 'Cf'（格式控制符）的字符。
    等价 Java: s = result.replaceAll("\\p{Cf}", "")
    """
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")


def is_nullish(result: Optional[str]) -> bool:
    """
    判断字符串是否为“空值”：
    null / 仅空白 / "null" / "null" + 仅分隔符或换行符
    保持与 Java 版本完全一致的判断逻辑与字符集合。
    """
    if result is None:
        return True

    # 去掉不可见格式化字符（如 BOM、零宽字符等），再 trim 空白
    s = _remove_format_chars(result).strip()
    if len(s) == 0:
        return True

    # 允许作为“分隔符/尾随噪声”的字符集合：
    # 空白(\s)、英文/中文逗号、分号、竖线、斜杠、反斜杠、下划线、短横、冒号（含中文）
    tail_separators_pattern = r'[\s,，;；|｜/\\_\-:：]+$'

    # 把尾部“分隔符/换行符”全部剔除，再做大小写不敏感比较
    stripped = re.sub(tail_separators_pattern, "", s)
    return stripped.lower() == "null"


def send_task(TASK_KEY: str, task_content: str) -> str:
    """
    向云端发送一个任务并返回处理结果（自动处理心跳、错误提示与Base64解码）。
    ✅ 新增：设置60秒总超时限制，防止客户端永久阻塞等待
    :param TASK_KEY: 任务密钥（可动态设置）
    :param task_content: 要发送的原始任务内容（支持多行）
    :return: 云端返回的处理结果（已自动处理各种异常和编码）
    """
    sock = None
    writer = None
    reader = None
    try:
        # ================== 1. 构造任务 ==================
        # 包装格式为：<Key:XXX>任务正文
        payload = f"<Key:{TASK_KEY}>{task_content}"

        # 将任务整体使用 Base64 编码，便于跨行传输
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        # ================== 2. 建立连接 ==================
        # 与 Java 保持一致：不为“连接”显式设置超时，仅为“读”设置 60s 超时
        sock = socket.create_connection((CLOUD_IP, CLOUD_PORT))
        sock.settimeout(60.0)  # ✅ 设置读取数据的最大超时时间为60秒，防止云端无响应时阻塞

        # 创建输入输出流，用于发送和接收数据（文本行）
        writer = sock.makefile(mode="w", encoding="utf-8", newline="\n")
        reader = sock.makefile(mode="r", encoding="utf-8", newline="\n")

        # 发送任务（每条任务以换行结束，云端以 readline() 接收）
        writer.write(encoded_payload + "\n")
        writer.flush()

        # ================== 3. 读取响应 ==================
        result_parts = []
        has_heartbeat = False  # 标记是否收到过云端的心跳包

        # 持续读取云端返回的每一行数据，直到连接关闭或超时
        for line in reader:
            line = line.rstrip("\n")

            if "<HeartbeatTest>" in line:
                # 云端心跳包，跳过不处理
                has_heartbeat = True
                continue

            # 判断是否是云端返回的错误/提示信息（这些信息是明文，不进行 Base64 解码）
            if line.startswith("[ERROR]") or line.startswith("[WARN]") or line.startswith("[INFO]"):
                result_parts.append(line + "\n")
                continue

            # 否则认为是 Base64 编码后的服务端执行结果，尝试进行解码
            try:
                decoded = base64.b64decode(line).decode("utf-8")
                result_parts.append(decoded + "\n")
            except Exception:
                # 若解码失败，说明数据格式可能异常，记录原文
                result_parts.append("[ERROR]无法解析返回内容（可能非标准Base64编码）:" + line + "\n")

        # ================== 4. 空结果处理 ==================
        if not result_parts:
            if has_heartbeat:
                return "[ERROR]连接正常但未收到任何实际任务结果"
            else:
                return "[ERROR]未收到任何响应，连接可能提前断开"

        return "".join(result_parts).rstrip("\n")  # 去除末尾换行，返回最终结果

    except socket.timeout:
        # ✅ 捕获Socket超时异常（读超时触发），提示用户
        return "[ERROR]等待云端返回超时（超过60秒），请检查云端或服务端状态"
    except OSError as e:
        # 网络连接或通信发生异常
        return "[ERROR]连接或通信异常：" + str(e)
    finally:
        # ========== 5. 安全释放资源 ==========
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass


def print_cloud_responses_help() -> None:
    """
    提示：列出云端所有可能返回的响应信息及其含义。
    与 Java 版本逐行一致。
    """
    print("======= 云端可能返回的响应及含义说明 =======")
    print("[ERROR]任务派发失败（收到空任务）       → 任务内容为空或格式错误")
    print("[ERROR]未找到任何可用服务端...         → 密钥无效、无服务在线或服务正忙")
    print("[ERROR]服务端处理超时或异常             → 服务端120秒内未返回结果")
    print("[ERROR]云端内部处理异常，请稍后再试     → 云端写入客户端失败（连接中断）")
    print("[ERROR]连接正常但未收到任何实际任务结果 → 可能云端响应中只包含心跳包，无结果")
    print("[ERROR]未收到任何响应，连接可能断开     → 连接建立但未有任何数据返回")
    print("[ERROR]等待云端返回超时（超过60秒）     → 本地Socket超时断开（新增）")
    print("<HeartbeatTest>（自动忽略）            → 云端发送的心跳保活信息")
    print("其余任何内容                            → 均为服务端实际处理结果")
    print("===========================================")


def api(TASK_KEY: str, task_content: str) -> Optional[str]:
    """
    等价 Java: public static String api(String TASK_KEY, String taskContent)
    - 重试 3 次
    - 若 isNullish(output) 则继续重试；否则立即返回
    - 全部失败返回 None
    """
    # Java 原文：const int retry = 3;  // Python 采用同义常量
    RETRY = 3

    for _ in range(RETRY):
        output = send_task(TASK_KEY, task_content)

        if is_nullish(output):
            # 若返回空字串（或“null”等价），则尝试重发
            continue
        else:
            # 否则返回结果
            return output

    # 若多次重发依旧得到失败结果，则返回空（Python 对齐 Java 的 null → None）
    return None


def _can_reach(host: str, port: int, timeout: float = 2.0) -> bool:
    """快速探测某个 host:port 是否可达（TCP 三次握手），成功即认为可连通。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def wait_for_network(
    primary_host: str,
    primary_port: int,
    check_hosts: Tuple[Tuple[str, int], ...] = (("1.1.1.1", 53), ("8.8.8.8", 53)),
    check_timeout: float = 2.0,
    init_sleep: float = 1.0,
    max_sleep: float = 10.0,
) -> None:
    """
    等待网络恢复：
    - 先测公共 DNS (1.1.1.1:53 / 8.8.8.8:53)，两者都不通 → 视为“本机离线”
    - 任一公共 DNS 可达 → 认为本机在线，此时不阻塞（即便云端暂不可达，也交由 send_task 自己重试/报错）
    - 指数退避休眠，直到网络恢复
    """
    sleep_s = init_sleep
    while True:
        # A. 快速判断是否“本机离线”（两个公共探测都失败才算离线）
        dns_ok = any(_can_reach(h, p, check_timeout) for h, p in check_hosts)
        if dns_ok:
            # 本机在线，直接返回，交由后续对云端的发送逻辑处理
            return

        # B.（可选）补充对目标云端的直连探测——如果能打通，也算网络恢复
        if _can_reach(primary_host, primary_port, check_timeout):
            return

        # C. 否则：仍离线 → 等待后继续探测
        print(f"[WARN] 未检测到网络连通，{sleep_s:.1f}s 后重试…")
        time.sleep(sleep_s)
        sleep_s = min(max_sleep, sleep_s * 1.7)


if __name__ == "__main__":
    # print_cloud_responses_help()  # 打印提示（可选）

    # === 网络连通测试【若不需要可自行删除】 ===
    wait_for_network(CLOUD_IP, CLOUD_PORT)
    backoff = 1.0
    while not _can_reach(CLOUD_IP, CLOUD_PORT, 2.0):
        print(f"[WARN] 目标 {CLOUD_IP}:{CLOUD_PORT} 未连通，{backoff:.1f}s 后重试…")
        time.sleep(backoff)
        backoff = min(10.0, backoff * 1.7)

    # === 对话接口 ===
    key = "TnumU6cM" #服务密钥
    command = "[create_conversation=true]" # 新建对话指令（若想避免历史信息干扰，则每次都把这个command加入到query中）【注意：至少每5-10次query就新建一次对话，以此规避大模型厂商的反扒措施，另外格式不要变，不会影响你的提问，这条指令是给爬虫系统看的，收到这条指令就会在后台新建对话】
    query = "请问最近国际上有什么大事发生？\n请给出一周以内的信息。" # query（举例）

    # laska 10.27测试用模型进行逻辑表达式的生成和推理
    instruction = "You are a logical reasoning agent. Your task is to solve PrOntoQA logical reasoning questions using strict logic, not intuition. You must (1) translate the natural language statements into formal logical expressions, (2) perform step-by-step symbolic reasoning, and (3) give the final answer. Follow the required output format exactly."
    query = "Problem:\nEach jompus is fruity. Every jompus is a wumpus. Every wumpus is not transparent. Wumpuses are tumpuses. Tumpuses are mean. Tumpuses are vumpuses. Every vumpus is cold. Each vumpus is a yumpus. Yumpuses are orange. Yumpuses are numpuses. Numpuses are dull. Each numpus is a dumpus. Every dumpus is not shy. Impuses are shy. Dumpuses are rompuses. Each rompus is liquid. Rompuses are zumpuses. Alex is a tumpus.\nQuestion:\nTrue or false: Alex is not shy."
    if instruction != "":
        query = instruction + "\n\n" + query

    result = api(key, query + command) # 函数api()第一个参数为密钥，第二个参数为发送数据
    print("\n[收到结果]：\n{}".format(result))
