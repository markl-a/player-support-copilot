from copilot.fields import extract, luhn_ok, mask
from copilot.guard import check


def test_extracts_order_ids_member_table_device_next_to_chinese():
    f = extract("訂單GPA.1234-5678-9012-34567和123456789012345 會員編號：12345678 T1234那桌 iPhone 15")
    assert f["order_ids"] == ["GPA.1234-5678-9012-34567", "123456789012345"]
    assert f["member_id"] == "12345678" and f["table"] == "T1234" and f["device"] == "iPhone 15"


def test_mask_hides_contact_and_card_but_keeps_order_numbers():
    card = "4111 1111 1111 1111"                      # Luhn-valid test card
    apple = "123456789012345"
    text = f"手機0912345678 信箱a.b@gmail.com 身分證A123456789 卡號{card} 訂單{apple}"
    out = mask(text, keep={apple})
    assert "0912345678" not in out and "a.b@gmail.com" not in out and "A123456789" not in out
    assert "<card>" in out and apple in out


def test_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")


def test_guard_blocks_unsafe_drafts():
    cites = ["refund.md:3"]
    assert check("請提供您的密碼以便查詢", cites)
    assert check("我們保證全額退款", cites)
    assert check("將補償您 10000 點", cites)
    assert check("對方已被停權，謝謝檢舉", cites)
    assert check("請聯絡 0912345678", cites)
    assert check("退款需向平台申請", []) == ["no policy citation"]
    assert check("退款需向 Google Play 申請，已使用的點數無法退款。", cites) == []
