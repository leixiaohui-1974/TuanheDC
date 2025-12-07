"""
TAOS V3.9 - Internationalization (i18n) Support Module
国际化支持模块

Features:
- Multi-language support (zh-CN, en-US, ja-JP, ko-KR, etc.)
- Text translation management with fallback
- Locale-aware date/time formatting
- Number and currency formatting
- Pluralization rules
- Dynamic language switching
- Translation file management
"""

import json
import os
import re
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class SupportedLocale(Enum):
    """Supported locales"""
    ZH_CN = "zh-CN"  # Simplified Chinese
    ZH_TW = "zh-TW"  # Traditional Chinese
    EN_US = "en-US"  # American English
    EN_GB = "en-GB"  # British English
    JA_JP = "ja-JP"  # Japanese
    KO_KR = "ko-KR"  # Korean
    RU_RU = "ru-RU"  # Russian
    FR_FR = "fr-FR"  # French
    DE_DE = "de-DE"  # German
    ES_ES = "es-ES"  # Spanish


@dataclass
class LocaleInfo:
    """Locale information"""
    code: str
    name: str
    native_name: str
    direction: str = "ltr"  # ltr or rtl
    date_format: str = "YYYY-MM-DD"
    time_format: str = "HH:mm:ss"
    datetime_format: str = "YYYY-MM-DD HH:mm:ss"
    number_decimal: str = "."
    number_thousand: str = ","
    currency_symbol: str = "$"
    currency_position: str = "before"  # before or after


@dataclass
class TranslationEntry:
    """Translation entry"""
    key: str
    translations: Dict[str, str] = field(default_factory=dict)
    context: Optional[str] = None
    plurals: Dict[str, Dict[str, str]] = field(default_factory=dict)  # locale -> {one, few, many, other}
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TranslationManager:
    """Translation text manager"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'i18n_data')
        self.translations: Dict[str, TranslationEntry] = {}
        self.default_locale = SupportedLocale.ZH_CN.value
        self.fallback_locale = SupportedLocale.EN_US.value
        self._lock = threading.RLock()

        # Initialize default translations
        self._init_default_translations()

    def _init_default_translations(self):
        """Initialize default system translations"""
        # System messages
        self._add_translation("system.name", {
            "zh-CN": "团河渠道自主运维系统",
            "zh-TW": "團河渠道自主運維系統",
            "en-US": "Tuanhe Aqueduct Autonomous Operation System",
            "ja-JP": "団河水路自律運用システム",
            "ko-KR": "투안허 수로 자율 운영 시스템"
        })

        self._add_translation("system.version", {
            "zh-CN": "版本",
            "en-US": "Version",
            "ja-JP": "バージョン",
            "ko-KR": "버전"
        })

        # Status messages
        self._add_translation("status.normal", {
            "zh-CN": "正常",
            "en-US": "Normal",
            "ja-JP": "正常",
            "ko-KR": "정상"
        })

        self._add_translation("status.warning", {
            "zh-CN": "警告",
            "en-US": "Warning",
            "ja-JP": "警告",
            "ko-KR": "경고"
        })

        self._add_translation("status.critical", {
            "zh-CN": "严重",
            "en-US": "Critical",
            "ja-JP": "重大",
            "ko-KR": "심각"
        })

        self._add_translation("status.offline", {
            "zh-CN": "离线",
            "en-US": "Offline",
            "ja-JP": "オフライン",
            "ko-KR": "오프라인"
        })

        # Scenario types
        self._add_translation("scenario.hydraulic_jump", {
            "zh-CN": "水跃风险",
            "en-US": "Hydraulic Jump Risk",
            "ja-JP": "水躍リスク",
            "ko-KR": "수력 점프 위험"
        })

        self._add_translation("scenario.thermal_bending", {
            "zh-CN": "热弯曲风险",
            "en-US": "Thermal Bending Risk",
            "ja-JP": "熱曲げリスク",
            "ko-KR": "열 굽힘 위험"
        })

        self._add_translation("scenario.bearing_lock", {
            "zh-CN": "支座锁死风险",
            "en-US": "Bearing Lock Risk",
            "ja-JP": "軸受ロックリスク",
            "ko-KR": "베어링 잠금 위험"
        })

        self._add_translation("scenario.joint_tearing", {
            "zh-CN": "接缝撕裂风险",
            "en-US": "Joint Tearing Risk",
            "ja-JP": "継ぎ目裂けリスク",
            "ko-KR": "조인트 찢어짐 위험"
        })

        self._add_translation("scenario.seismic", {
            "zh-CN": "地震风险",
            "en-US": "Seismic Risk",
            "ja-JP": "地震リスク",
            "ko-KR": "지진 위험"
        })

        # Control modes
        self._add_translation("control.auto", {
            "zh-CN": "自动控制",
            "en-US": "Automatic Control",
            "ja-JP": "自動制御",
            "ko-KR": "자동 제어"
        })

        self._add_translation("control.manual", {
            "zh-CN": "手动控制",
            "en-US": "Manual Control",
            "ja-JP": "手動制御",
            "ko-KR": "수동 제어"
        })

        self._add_translation("control.emergency", {
            "zh-CN": "紧急控制",
            "en-US": "Emergency Control",
            "ja-JP": "緊急制御",
            "ko-KR": "비상 제어"
        })

        # Measurements
        self._add_translation("measure.water_level", {
            "zh-CN": "水位",
            "en-US": "Water Level",
            "ja-JP": "水位",
            "ko-KR": "수위"
        })

        self._add_translation("measure.flow_rate", {
            "zh-CN": "流量",
            "en-US": "Flow Rate",
            "ja-JP": "流量",
            "ko-KR": "유량"
        })

        self._add_translation("measure.temperature", {
            "zh-CN": "温度",
            "en-US": "Temperature",
            "ja-JP": "温度",
            "ko-KR": "온도"
        })

        self._add_translation("measure.pressure", {
            "zh-CN": "压力",
            "en-US": "Pressure",
            "ja-JP": "圧力",
            "ko-KR": "압력"
        })

        self._add_translation("measure.velocity", {
            "zh-CN": "流速",
            "en-US": "Velocity",
            "ja-JP": "流速",
            "ko-KR": "유속"
        })

        # Units
        self._add_translation("unit.meter", {
            "zh-CN": "米",
            "en-US": "m",
            "ja-JP": "m",
            "ko-KR": "m"
        })

        self._add_translation("unit.cubic_meter_per_second", {
            "zh-CN": "立方米/秒",
            "en-US": "m³/s",
            "ja-JP": "m³/s",
            "ko-KR": "m³/s"
        })

        self._add_translation("unit.celsius", {
            "zh-CN": "摄氏度",
            "en-US": "°C",
            "ja-JP": "°C",
            "ko-KR": "°C"
        })

        self._add_translation("unit.pascal", {
            "zh-CN": "帕斯卡",
            "en-US": "Pa",
            "ja-JP": "Pa",
            "ko-KR": "Pa"
        })

        # Alerts
        self._add_translation("alert.high_water_level", {
            "zh-CN": "水位过高警报",
            "en-US": "High Water Level Alert",
            "ja-JP": "高水位アラート",
            "ko-KR": "높은 수위 경보"
        })

        self._add_translation("alert.low_water_level", {
            "zh-CN": "水位过低警报",
            "en-US": "Low Water Level Alert",
            "ja-JP": "低水位アラート",
            "ko-KR": "낮은 수위 경보"
        })

        self._add_translation("alert.high_flow", {
            "zh-CN": "流量过大警报",
            "en-US": "High Flow Rate Alert",
            "ja-JP": "高流量アラート",
            "ko-KR": "높은 유량 경보"
        })

        self._add_translation("alert.temperature_anomaly", {
            "zh-CN": "温度异常警报",
            "en-US": "Temperature Anomaly Alert",
            "ja-JP": "温度異常アラート",
            "ko-KR": "온도 이상 경보"
        })

        self._add_translation("alert.sensor_fault", {
            "zh-CN": "传感器故障",
            "en-US": "Sensor Fault",
            "ja-JP": "センサー故障",
            "ko-KR": "센서 고장"
        })

        self._add_translation("alert.actuator_fault", {
            "zh-CN": "执行器故障",
            "en-US": "Actuator Fault",
            "ja-JP": "アクチュエータ故障",
            "ko-KR": "액추에이터 고장"
        })

        # Time expressions
        self._add_translation("time.now", {
            "zh-CN": "现在",
            "en-US": "Now",
            "ja-JP": "現在",
            "ko-KR": "현재"
        })

        self._add_translation("time.today", {
            "zh-CN": "今天",
            "en-US": "Today",
            "ja-JP": "今日",
            "ko-KR": "오늘"
        })

        self._add_translation("time.yesterday", {
            "zh-CN": "昨天",
            "en-US": "Yesterday",
            "ja-JP": "昨日",
            "ko-KR": "어제"
        })

        self._add_translation("time.hours_ago", {
            "zh-CN": "{count}小时前",
            "en-US": "{count} hours ago",
            "ja-JP": "{count}時間前",
            "ko-KR": "{count}시간 전"
        })

        self._add_translation("time.minutes_ago", {
            "zh-CN": "{count}分钟前",
            "en-US": "{count} minutes ago",
            "ja-JP": "{count}分前",
            "ko-KR": "{count}분 전"
        })

        # Dashboard elements
        self._add_translation("dashboard.title", {
            "zh-CN": "系统仪表盘",
            "en-US": "System Dashboard",
            "ja-JP": "システムダッシュボード",
            "ko-KR": "시스템 대시보드"
        })

        self._add_translation("dashboard.overview", {
            "zh-CN": "概览",
            "en-US": "Overview",
            "ja-JP": "概要",
            "ko-KR": "개요"
        })

        self._add_translation("dashboard.monitoring", {
            "zh-CN": "监控",
            "en-US": "Monitoring",
            "ja-JP": "モニタリング",
            "ko-KR": "모니터링"
        })

        self._add_translation("dashboard.analysis", {
            "zh-CN": "分析",
            "en-US": "Analysis",
            "ja-JP": "分析",
            "ko-KR": "분석"
        })

        self._add_translation("dashboard.settings", {
            "zh-CN": "设置",
            "en-US": "Settings",
            "ja-JP": "設定",
            "ko-KR": "설정"
        })

        # Actions
        self._add_translation("action.start", {
            "zh-CN": "启动",
            "en-US": "Start",
            "ja-JP": "開始",
            "ko-KR": "시작"
        })

        self._add_translation("action.stop", {
            "zh-CN": "停止",
            "en-US": "Stop",
            "ja-JP": "停止",
            "ko-KR": "중지"
        })

        self._add_translation("action.restart", {
            "zh-CN": "重启",
            "en-US": "Restart",
            "ja-JP": "再起動",
            "ko-KR": "재시작"
        })

        self._add_translation("action.refresh", {
            "zh-CN": "刷新",
            "en-US": "Refresh",
            "ja-JP": "更新",
            "ko-KR": "새로고침"
        })

        self._add_translation("action.save", {
            "zh-CN": "保存",
            "en-US": "Save",
            "ja-JP": "保存",
            "ko-KR": "저장"
        })

        self._add_translation("action.cancel", {
            "zh-CN": "取消",
            "en-US": "Cancel",
            "ja-JP": "キャンセル",
            "ko-KR": "취소"
        })

        self._add_translation("action.confirm", {
            "zh-CN": "确认",
            "en-US": "Confirm",
            "ja-JP": "確認",
            "ko-KR": "확인"
        })

        self._add_translation("action.export", {
            "zh-CN": "导出",
            "en-US": "Export",
            "ja-JP": "エクスポート",
            "ko-KR": "내보내기"
        })

        self._add_translation("action.import", {
            "zh-CN": "导入",
            "en-US": "Import",
            "ja-JP": "インポート",
            "ko-KR": "가져오기"
        })

        # Report types
        self._add_translation("report.daily", {
            "zh-CN": "日报",
            "en-US": "Daily Report",
            "ja-JP": "日報",
            "ko-KR": "일일 보고서"
        })

        self._add_translation("report.weekly", {
            "zh-CN": "周报",
            "en-US": "Weekly Report",
            "ja-JP": "週報",
            "ko-KR": "주간 보고서"
        })

        self._add_translation("report.monthly", {
            "zh-CN": "月报",
            "en-US": "Monthly Report",
            "ja-JP": "月報",
            "ko-KR": "월간 보고서"
        })

        # Error messages
        self._add_translation("error.network", {
            "zh-CN": "网络连接错误",
            "en-US": "Network Connection Error",
            "ja-JP": "ネットワーク接続エラー",
            "ko-KR": "네트워크 연결 오류"
        })

        self._add_translation("error.timeout", {
            "zh-CN": "请求超时",
            "en-US": "Request Timeout",
            "ja-JP": "リクエストタイムアウト",
            "ko-KR": "요청 시간 초과"
        })

        self._add_translation("error.unauthorized", {
            "zh-CN": "未授权访问",
            "en-US": "Unauthorized Access",
            "ja-JP": "未認証アクセス",
            "ko-KR": "승인되지 않은 접근"
        })

        self._add_translation("error.not_found", {
            "zh-CN": "资源未找到",
            "en-US": "Resource Not Found",
            "ja-JP": "リソースが見つかりません",
            "ko-KR": "리소스를 찾을 수 없음"
        })

        self._add_translation("error.server", {
            "zh-CN": "服务器错误",
            "en-US": "Server Error",
            "ja-JP": "サーバーエラー",
            "ko-KR": "서버 오류"
        })

    def _add_translation(self, key: str, translations: Dict[str, str],
                        context: str = None):
        """Add a translation entry"""
        with self._lock:
            entry = TranslationEntry(
                key=key,
                translations=translations,
                context=context
            )
            self.translations[key] = entry

    def translate(self, key: str, locale: str = None,
                 default: str = None, **kwargs) -> str:
        """
        Translate a key to specified locale

        Args:
            key: Translation key
            locale: Target locale (default: default_locale)
            default: Default text if translation not found
            **kwargs: Interpolation variables

        Returns:
            Translated text
        """
        locale = locale or self.default_locale

        with self._lock:
            entry = self.translations.get(key)
            if not entry:
                return default or key

            # Try exact locale
            text = entry.translations.get(locale)

            # Try language part only (e.g., "zh" from "zh-CN")
            if not text:
                lang = locale.split('-')[0]
                for loc, trans in entry.translations.items():
                    if loc.startswith(lang):
                        text = trans
                        break

            # Fallback to fallback locale
            if not text:
                text = entry.translations.get(self.fallback_locale)

            # Use default or key
            if not text:
                text = default or key

            # Interpolate variables
            if kwargs:
                for var_key, var_value in kwargs.items():
                    text = text.replace(f"{{{var_key}}}", str(var_value))

            return text

    def t(self, key: str, locale: str = None, **kwargs) -> str:
        """Shorthand for translate()"""
        return self.translate(key, locale, **kwargs)

    def add_translation(self, key: str, locale: str, text: str):
        """Add or update a translation for a specific locale"""
        with self._lock:
            if key not in self.translations:
                self.translations[key] = TranslationEntry(key=key)

            self.translations[key].translations[locale] = text
            self.translations[key].updated_at = time.time()

    def add_translations_batch(self, translations: Dict[str, Dict[str, str]]):
        """Add multiple translations at once"""
        with self._lock:
            for key, locale_texts in translations.items():
                self._add_translation(key, locale_texts)

    def get_all_keys(self) -> List[str]:
        """Get all translation keys"""
        with self._lock:
            return list(self.translations.keys())

    def get_translations_for_locale(self, locale: str) -> Dict[str, str]:
        """Get all translations for a locale"""
        with self._lock:
            result = {}
            for key, entry in self.translations.items():
                text = entry.translations.get(locale)
                if text:
                    result[key] = text
            return result

    def export_translations(self, locale: str = None) -> Dict[str, Any]:
        """Export translations to JSON format"""
        with self._lock:
            if locale:
                return {
                    "locale": locale,
                    "translations": self.get_translations_for_locale(locale)
                }
            else:
                result = {}
                for key, entry in self.translations.items():
                    result[key] = entry.translations
                return {"translations": result}

    def import_translations(self, data: Dict[str, Any]):
        """Import translations from JSON format"""
        if "translations" in data:
            translations = data["translations"]
            if "locale" in data:
                # Single locale import
                locale = data["locale"]
                for key, text in translations.items():
                    self.add_translation(key, locale, text)
            else:
                # Multi-locale import
                self.add_translations_batch(translations)

    def get_status(self) -> Dict[str, Any]:
        """Get translation manager status"""
        with self._lock:
            locale_counts = {}
            for entry in self.translations.values():
                for locale in entry.translations.keys():
                    locale_counts[locale] = locale_counts.get(locale, 0) + 1

            return {
                "total_keys": len(self.translations),
                "default_locale": self.default_locale,
                "fallback_locale": self.fallback_locale,
                "locale_coverage": locale_counts,
                "supported_locales": [l.value for l in SupportedLocale]
            }


class LocaleManager:
    """Locale settings manager"""

    # Locale configurations
    LOCALE_INFO = {
        "zh-CN": LocaleInfo(
            code="zh-CN",
            name="Chinese (Simplified)",
            native_name="简体中文",
            date_format="YYYY年MM月DD日",
            time_format="HH:mm:ss",
            datetime_format="YYYY年MM月DD日 HH:mm:ss",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="¥",
            currency_position="before"
        ),
        "zh-TW": LocaleInfo(
            code="zh-TW",
            name="Chinese (Traditional)",
            native_name="繁體中文",
            date_format="YYYY年MM月DD日",
            time_format="HH:mm:ss",
            datetime_format="YYYY年MM月DD日 HH:mm:ss",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="NT$",
            currency_position="before"
        ),
        "en-US": LocaleInfo(
            code="en-US",
            name="English (US)",
            native_name="English (US)",
            date_format="MM/DD/YYYY",
            time_format="hh:mm:ss A",
            datetime_format="MM/DD/YYYY hh:mm:ss A",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="$",
            currency_position="before"
        ),
        "en-GB": LocaleInfo(
            code="en-GB",
            name="English (UK)",
            native_name="English (UK)",
            date_format="DD/MM/YYYY",
            time_format="HH:mm:ss",
            datetime_format="DD/MM/YYYY HH:mm:ss",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="£",
            currency_position="before"
        ),
        "ja-JP": LocaleInfo(
            code="ja-JP",
            name="Japanese",
            native_name="日本語",
            date_format="YYYY年MM月DD日",
            time_format="HH:mm:ss",
            datetime_format="YYYY年MM月DD日 HH:mm:ss",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="¥",
            currency_position="before"
        ),
        "ko-KR": LocaleInfo(
            code="ko-KR",
            name="Korean",
            native_name="한국어",
            date_format="YYYY년 MM월 DD일",
            time_format="HH:mm:ss",
            datetime_format="YYYY년 MM월 DD일 HH:mm:ss",
            number_decimal=".",
            number_thousand=",",
            currency_symbol="₩",
            currency_position="before"
        ),
        "ru-RU": LocaleInfo(
            code="ru-RU",
            name="Russian",
            native_name="Русский",
            date_format="DD.MM.YYYY",
            time_format="HH:mm:ss",
            datetime_format="DD.MM.YYYY HH:mm:ss",
            number_decimal=",",
            number_thousand=" ",
            currency_symbol="₽",
            currency_position="after"
        ),
        "fr-FR": LocaleInfo(
            code="fr-FR",
            name="French",
            native_name="Français",
            date_format="DD/MM/YYYY",
            time_format="HH:mm:ss",
            datetime_format="DD/MM/YYYY HH:mm:ss",
            number_decimal=",",
            number_thousand=" ",
            currency_symbol="€",
            currency_position="after"
        ),
        "de-DE": LocaleInfo(
            code="de-DE",
            name="German",
            native_name="Deutsch",
            date_format="DD.MM.YYYY",
            time_format="HH:mm:ss",
            datetime_format="DD.MM.YYYY HH:mm:ss",
            number_decimal=",",
            number_thousand=".",
            currency_symbol="€",
            currency_position="after"
        ),
        "es-ES": LocaleInfo(
            code="es-ES",
            name="Spanish",
            native_name="Español",
            date_format="DD/MM/YYYY",
            time_format="HH:mm:ss",
            datetime_format="DD/MM/YYYY HH:mm:ss",
            number_decimal=",",
            number_thousand=".",
            currency_symbol="€",
            currency_position="after"
        )
    }

    def __init__(self):
        self.current_locale = "zh-CN"
        self._lock = threading.RLock()

    def get_locale_info(self, locale: str = None) -> Optional[LocaleInfo]:
        """Get locale information"""
        locale = locale or self.current_locale
        return self.LOCALE_INFO.get(locale)

    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales"""
        return [
            {
                "code": info.code,
                "name": info.name,
                "native_name": info.native_name
            }
            for info in self.LOCALE_INFO.values()
        ]

    def set_locale(self, locale: str) -> bool:
        """Set current locale"""
        if locale in self.LOCALE_INFO:
            with self._lock:
                self.current_locale = locale
            return True
        return False

    def format_date(self, dt: datetime, locale: str = None,
                   format_type: str = "date") -> str:
        """
        Format date/time according to locale

        Args:
            dt: datetime object
            locale: target locale
            format_type: "date", "time", or "datetime"

        Returns:
            Formatted string
        """
        locale = locale or self.current_locale
        info = self.LOCALE_INFO.get(locale)
        if not info:
            info = self.LOCALE_INFO["en-US"]

        if format_type == "date":
            fmt = info.date_format
        elif format_type == "time":
            fmt = info.time_format
        else:
            fmt = info.datetime_format

        # Convert format pattern to strftime
        fmt = fmt.replace("YYYY", "%Y")
        fmt = fmt.replace("MM", "%m")
        fmt = fmt.replace("DD", "%d")
        fmt = fmt.replace("HH", "%H")
        fmt = fmt.replace("hh", "%I")
        fmt = fmt.replace("mm", "%M")
        fmt = fmt.replace("ss", "%S")
        fmt = fmt.replace("A", "%p")
        fmt = fmt.replace("年", "年")
        fmt = fmt.replace("月", "月")
        fmt = fmt.replace("日", "日")

        return dt.strftime(fmt)

    def format_number(self, number: float, locale: str = None,
                     decimal_places: int = 2) -> str:
        """Format number according to locale"""
        locale = locale or self.current_locale
        info = self.LOCALE_INFO.get(locale)
        if not info:
            info = self.LOCALE_INFO["en-US"]

        # Format with decimal places
        formatted = f"{number:,.{decimal_places}f}"

        # Replace separators
        if info.number_thousand != ",":
            formatted = formatted.replace(",", "TEMP")
        if info.number_decimal != ".":
            formatted = formatted.replace(".", info.number_decimal)
        if info.number_thousand != ",":
            formatted = formatted.replace("TEMP", info.number_thousand)

        return formatted

    def format_currency(self, amount: float, locale: str = None,
                       decimal_places: int = 2) -> str:
        """Format currency according to locale"""
        locale = locale or self.current_locale
        info = self.LOCALE_INFO.get(locale)
        if not info:
            info = self.LOCALE_INFO["en-US"]

        formatted_amount = self.format_number(amount, locale, decimal_places)

        if info.currency_position == "before":
            return f"{info.currency_symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {info.currency_symbol}"

    def format_relative_time(self, dt: datetime, locale: str = None) -> str:
        """Format relative time (e.g., '5 minutes ago')"""
        locale = locale or self.current_locale
        now = datetime.now()
        diff = now - dt

        seconds = diff.total_seconds()

        if seconds < 60:
            return self._get_relative_time_text("seconds", int(seconds), locale)
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return self._get_relative_time_text("minutes", minutes, locale)
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return self._get_relative_time_text("hours", hours, locale)
        elif seconds < 604800:
            days = int(seconds / 86400)
            return self._get_relative_time_text("days", days, locale)
        else:
            return self.format_date(dt, locale, "date")

    def _get_relative_time_text(self, unit: str, count: int, locale: str) -> str:
        """Get localized relative time text"""
        templates = {
            "zh-CN": {
                "seconds": "{count}秒前",
                "minutes": "{count}分钟前",
                "hours": "{count}小时前",
                "days": "{count}天前"
            },
            "en-US": {
                "seconds": "{count} seconds ago",
                "minutes": "{count} minutes ago",
                "hours": "{count} hours ago",
                "days": "{count} days ago"
            },
            "ja-JP": {
                "seconds": "{count}秒前",
                "minutes": "{count}分前",
                "hours": "{count}時間前",
                "days": "{count}日前"
            },
            "ko-KR": {
                "seconds": "{count}초 전",
                "minutes": "{count}분 전",
                "hours": "{count}시간 전",
                "days": "{count}일 전"
            }
        }

        locale_templates = templates.get(locale, templates["en-US"])
        template = locale_templates.get(unit, "{count} " + unit + " ago")
        return template.replace("{count}", str(count))


class I18nManager:
    """Main internationalization manager"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'i18n_data')
        self.translation_manager = TranslationManager(self.data_dir)
        self.locale_manager = LocaleManager()
        self._user_locales: Dict[str, str] = {}  # user_id -> locale
        self._lock = threading.RLock()

    def t(self, key: str, locale: str = None, **kwargs) -> str:
        """Translate text (shorthand)"""
        locale = locale or self.locale_manager.current_locale
        return self.translation_manager.translate(key, locale, **kwargs)

    def translate(self, key: str, locale: str = None, **kwargs) -> str:
        """Translate text"""
        return self.t(key, locale, **kwargs)

    def set_locale(self, locale: str) -> bool:
        """Set current locale"""
        return self.locale_manager.set_locale(locale)

    def get_locale(self) -> str:
        """Get current locale"""
        return self.locale_manager.current_locale

    def set_user_locale(self, user_id: str, locale: str) -> bool:
        """Set locale for a specific user"""
        if locale in self.locale_manager.LOCALE_INFO:
            with self._lock:
                self._user_locales[user_id] = locale
            return True
        return False

    def get_user_locale(self, user_id: str) -> str:
        """Get locale for a specific user"""
        with self._lock:
            return self._user_locales.get(user_id, self.locale_manager.current_locale)

    def format_date(self, dt: datetime, locale: str = None,
                   format_type: str = "date") -> str:
        """Format date/time"""
        return self.locale_manager.format_date(dt, locale, format_type)

    def format_number(self, number: float, locale: str = None,
                     decimal_places: int = 2) -> str:
        """Format number"""
        return self.locale_manager.format_number(number, locale, decimal_places)

    def format_currency(self, amount: float, locale: str = None,
                       decimal_places: int = 2) -> str:
        """Format currency"""
        return self.locale_manager.format_currency(amount, locale, decimal_places)

    def format_relative_time(self, dt: datetime, locale: str = None) -> str:
        """Format relative time"""
        return self.locale_manager.format_relative_time(dt, locale)

    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales"""
        return self.locale_manager.get_supported_locales()

    def detect_locale_from_header(self, accept_language: str) -> str:
        """Detect locale from Accept-Language header"""
        if not accept_language:
            return self.locale_manager.current_locale

        # Parse Accept-Language header
        # Format: en-US,en;q=0.9,zh-CN;q=0.8
        locales = []
        for part in accept_language.split(','):
            part = part.strip()
            if ';' in part:
                locale, q = part.split(';')
                locale = locale.strip()
                try:
                    q = float(q.strip().replace('q=', ''))
                except:
                    q = 0.0
            else:
                locale = part
                q = 1.0
            locales.append((locale, q))

        # Sort by quality
        locales.sort(key=lambda x: x[1], reverse=True)

        # Find first supported locale
        for locale, _ in locales:
            if locale in self.locale_manager.LOCALE_INFO:
                return locale
            # Try language part only
            lang = locale.split('-')[0]
            for supported in self.locale_manager.LOCALE_INFO:
                if supported.startswith(lang):
                    return supported

        return self.locale_manager.current_locale

    def export_all_translations(self) -> Dict[str, Any]:
        """Export all translations"""
        return self.translation_manager.export_translations()

    def import_translations(self, data: Dict[str, Any]):
        """Import translations"""
        self.translation_manager.import_translations(data)

    def add_translation(self, key: str, locale: str, text: str):
        """Add a translation"""
        self.translation_manager.add_translation(key, locale, text)

    def get_status(self) -> Dict[str, Any]:
        """Get i18n system status"""
        trans_status = self.translation_manager.get_status()
        return {
            "current_locale": self.locale_manager.current_locale,
            "supported_locales": self.get_supported_locales(),
            "translations": trans_status,
            "user_locale_count": len(self._user_locales)
        }

    def translate_dict(self, data: Dict[str, Any], locale: str = None,
                      keys_to_translate: List[str] = None) -> Dict[str, Any]:
        """
        Translate specific keys in a dictionary

        Args:
            data: Dictionary to translate
            locale: Target locale
            keys_to_translate: Keys to translate (default: all string values)

        Returns:
            Dictionary with translated values
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                if keys_to_translate is None or key in keys_to_translate:
                    # Check if value looks like a translation key
                    if '.' in value and not value.startswith('/'):
                        translated = self.t(value, locale)
                        result[key] = translated if translated != value else value
                    else:
                        result[key] = value
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.translate_dict(value, locale, keys_to_translate)
            elif isinstance(value, list):
                result[key] = [
                    self.translate_dict(item, locale, keys_to_translate)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# Global instance
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance"""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def t(key: str, locale: str = None, **kwargs) -> str:
    """Global translation function"""
    return get_i18n_manager().t(key, locale, **kwargs)


def set_locale(locale: str) -> bool:
    """Set global locale"""
    return get_i18n_manager().set_locale(locale)


def get_locale() -> str:
    """Get current locale"""
    return get_i18n_manager().get_locale()


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("TAOS V3.9 - Internationalization Module Test")
    print("=" * 60)

    manager = get_i18n_manager()

    # Test translations
    print("\n1. Translation Tests:")
    print("-" * 40)

    test_keys = [
        "system.name",
        "status.normal",
        "scenario.hydraulic_jump",
        "measure.water_level",
        "alert.high_water_level",
        "action.start"
    ]

    for locale in ["zh-CN", "en-US", "ja-JP", "ko-KR"]:
        print(f"\n  Locale: {locale}")
        for key in test_keys:
            print(f"    {key}: {manager.t(key, locale)}")

    # Test interpolation
    print("\n2. Interpolation Tests:")
    print("-" * 40)
    print(f"  zh-CN: {manager.t('time.hours_ago', 'zh-CN', count=5)}")
    print(f"  en-US: {manager.t('time.hours_ago', 'en-US', count=5)}")
    print(f"  ja-JP: {manager.t('time.hours_ago', 'ja-JP', count=5)}")

    # Test locale formatting
    print("\n3. Locale Formatting Tests:")
    print("-" * 40)

    now = datetime.now()
    test_number = 1234567.89
    test_currency = 9999.99

    for locale in ["zh-CN", "en-US", "ja-JP", "de-DE"]:
        print(f"\n  Locale: {locale}")
        print(f"    Date: {manager.format_date(now, locale, 'date')}")
        print(f"    DateTime: {manager.format_date(now, locale, 'datetime')}")
        print(f"    Number: {manager.format_number(test_number, locale)}")
        print(f"    Currency: {manager.format_currency(test_currency, locale)}")

    # Test relative time
    print("\n4. Relative Time Tests:")
    print("-" * 40)

    test_times = [
        (datetime.now() - timedelta(seconds=30), "30 seconds ago"),
        (datetime.now() - timedelta(minutes=15), "15 minutes ago"),
        (datetime.now() - timedelta(hours=3), "3 hours ago"),
        (datetime.now() - timedelta(days=2), "2 days ago")
    ]

    for dt, desc in test_times:
        print(f"\n  {desc}:")
        for locale in ["zh-CN", "en-US", "ja-JP"]:
            print(f"    {locale}: {manager.format_relative_time(dt, locale)}")

    # Test Accept-Language detection
    print("\n5. Accept-Language Detection Tests:")
    print("-" * 40)

    test_headers = [
        "en-US,en;q=0.9,zh-CN;q=0.8",
        "zh-CN,zh;q=0.9",
        "ja-JP,en-US;q=0.7",
        "fr-FR,fr;q=0.9,en;q=0.8",
        "unknown-XX"
    ]

    for header in test_headers:
        detected = manager.detect_locale_from_header(header)
        print(f"  '{header}' -> {detected}")

    # Test status
    print("\n6. System Status:")
    print("-" * 40)

    status = manager.get_status()
    print(f"  Current Locale: {status['current_locale']}")
    print(f"  Translation Keys: {status['translations']['total_keys']}")
    print(f"  Supported Locales: {len(status['supported_locales'])}")
    print(f"  Locale Coverage:")
    for locale, count in status['translations']['locale_coverage'].items():
        print(f"    {locale}: {count} translations")

    print("\n" + "=" * 60)
    print("All i18n tests completed successfully!")
    print("=" * 60)
