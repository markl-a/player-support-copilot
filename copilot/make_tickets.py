"""Synthetic player-support tickets (Traditional Chinese) with a *phrasing split*.

Train and test use different sentence templates, so the classifier is scored on wording it has never seen, the
way new tickets arrive, instead of on paraphrases of its training data.
Writes data/train.jsonl, data/test.jsonl: {"text", "label", "order_id"|null}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

LABELS = ["topup_missing", "account_access", "disconnect_bug", "event_reward", "cheating_report", "refund_request", "other"]

# (train templates, test templates) per label. {oid} {amt} {game} {time} {dev} {tb} {nick} {ev}
T = {
    "topup_missing": ([
        "我剛剛儲值{amt}元但是點數沒有進來，訂單編號 {oid}",
        "付款成功了可是遊戲裡沒有金幣 {oid}",
        "儲值沒到帳！！已經過了一個小時 訂單{oid}",
        "{time}買了點數卡 帳戶沒增加",
        "扣款了但點數沒加 請幫我查 {oid}",
        "google play 有扣錢 遊戲裡的點數沒變",
    ], [
        "錢都付了東西呢？{oid}",
        "我的{amt}塊點數到底什麼時候會到 收據編號{oid}",
        "銀行通知已扣款 但是我的幣沒有增加",
        "買了禮包卻什麼都沒拿到 {oid}",
    ]),
    "account_access": ([
        "帳號登不進去一直顯示密碼錯誤",
        "我的帳號好像被盜了 裡面的幣都不見了",
        "換手機之後帳號就登入不了",
        "收不到簡訊驗證碼沒辦法登入",
        "有人登入我的帳號 改了我的暱稱",
        "用apple登入變成一個新帳號 舊的不見了",
    ], [
        "我是不是被駭了 金幣少了一大堆",
        "忘記當初用哪個方式綁定 現在進不去",
        "一直跳出帳號異常 無法進入遊戲",
        "帳號被別人拿去用了怎麼辦",
    ]),
    "disconnect_bug": ([
        "{game}打到一半斷線 回來牌局就結束了",
        "遊戲一直閃退 {dev}",
        "{time}的時候整個卡住 然後就被踢出房間",
        "連線一直中斷 其他app都正常",
        "胡牌的時候畫面當掉 結果沒算到",
        "更新之後開不起來 {dev}",
    ], [
        "打{game}時網路突然斷掉 回來已經被代打輸光了",
        "{dev}一進牌桌就黑畫面",
        "剛剛伺服器是不是有問題 大家都斷了",
        "畫面一直轉圈圈進不去牌局",
    ]),
    "event_reward": ([
        "{ev}活動的獎勵我沒有收到",
        "簽到滿七天 可是沒拿到獎勵",
        "活動任務完成了 信箱裡沒有東西",
        "{ev}排行榜的獎品什麼時候發",
        "兌換碼輸入之後顯示已使用 但我沒用過",
        "首儲禮沒有送",
    ], [
        "明明達成{ev}條件了 獎勵呢",
        "說好的登入禮都沒發",
        "任務進度卡在99% 領不到",
        "{ev}的限定頭像沒出現",
    ]),
    "cheating_report": ([
        "{tb}那桌的{nick}一直作弊 好像有看牌外掛",
        "我要檢舉{nick} 他跟隊友串通",
        "有人開外掛 每把都自摸",
        "{nick}在聊天室罵人還說他有外掛",
        "檢舉{tb}有兩個帳號同一個人在打",
        "對手出牌速度快到不正常 懷疑用程式",
    ], [
        "{nick}這個人絕對有問題 連續天胡",
        "{tb}有人在打假牌 請處理",
        "懷疑{nick}開小號互相送分",
        "遇到機器人在打 完全不用思考時間",
    ]),
    "refund_request": ([
        "我要退款 不小心買錯了",
        "小孩用我的手機亂儲值 可以退嗎",
        "{amt}元的禮包我不要了 請退錢",
        "申請退費 {oid}",
        "重複扣款兩次 要退一筆",
        "不想玩了 剩下的點數可以換回錢嗎",
    ], [
        "誤觸購買可以取消嗎 {oid}",
        "家裡長輩不知道會扣錢 想辦退款",
        "我要把錢拿回來",
        "買了之後發現不能用 要求退費",
    ]),
    "other": ([
        "請問你們有電腦版嗎",
        "暱稱可以改嗎 要花錢嗎",
        "想問下一次改版是什麼時候",
        "你們有在徵客服人員嗎",
        "建議增加好友房的功能",
        "遊戲規則的台數怎麼算",
    ], [
        "可以關掉背景音樂嗎",
        "有沒有新手教學",
        "想合作直播 要找誰",
        "我覺得配色很好看 繼續加油",
    ]),
}

FILL = {
    "amt": ["30", "90", "290", "890", "1490"],
    "game": ["麻將", "十六張", "德州撲克", "大老二", "象棋"],
    "time": ["昨天晚上", "今天早上十點", "剛剛", "9/27 21:05", "半小時前"],
    "dev": ["iPhone 15", "三星 A54", "小米 13", "iPad", "Pixel 8"],
    "tb": ["T1234", "第88桌", "T3021", "好友房 5566"],
    "nick": ["雀神阿明", "小白兔", "a123456", "胡牌王"],
    "ev": ["中秋", "週年慶", "新手七日", "連續登入"],
}
NOISE = ["", "", "", " 拜託快點", " 急!!!", " 謝謝", "\n我的會員編號是 {mid}", " 手機 09{ph}", " 我的email是 {em}", " 😡😡"]


def order_id(rng: random.Random) -> str:
    return rng.choice([
        "GPA.%04d-%04d-%04d-%05d" % tuple(rng.randrange(10 ** k) for k in (4, 4, 4, 5)),
        "%d" % rng.randrange(10 ** 14, 10 ** 15),                   # App Store order number
    ])


def render(tpl: str, rng: random.Random) -> tuple[str, str | None]:
    oid = order_id(rng) if "{oid}" in tpl else None
    vals = {k: rng.choice(v) for k, v in FILL.items()}
    text = tpl.format(oid=oid or "", **vals)
    noise = rng.choice(NOISE).format(mid=rng.randrange(10 ** 7, 10 ** 8), ph=rng.randrange(10 ** 7, 10 ** 8),
                                     em=f"player{rng.randrange(999)}@gmail.com")
    return (text + noise).strip(), oid


def main(n_per_label_train: int = 120, n_per_label_test: int = 40, seed: int = 11) -> None:
    rng = random.Random(seed)
    out = Path(__file__).resolve().parent.parent / "data"
    for split, n, idx in (("train", n_per_label_train, 0), ("test", n_per_label_test, 1)):
        rows = []
        for label in LABELS:
            for _ in range(n):
                text, oid = render(rng.choice(T[label][idx]), rng)
                rows.append({"text": text, "label": label, "order_id": oid})
        rng.shuffle(rows)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
