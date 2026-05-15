import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Modal } from "@agentscope-ai/design";
import {
  CheckCircleFilled,
  CopyOutlined,
  ExportOutlined,
  LoadingOutlined,
  LogoutOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import api from "../../../../../api";
import type {
  AuthStartResult,
  AuthStatusResult,
  ProviderInfo,
} from "../../../../../api/types";
import { useAppMessage } from "../../../../../hooks/useAppMessage";

interface DeviceCodeAuthPanelProps {
  provider: ProviderInfo;
  onAuthChanged: () => void;
}

function secondsToMs(seconds?: number | null): number {
  const normalized = Math.max(1, seconds ?? 5);
  return normalized * 1000;
}

export function DeviceCodeAuthPanel({
  provider,
  onAuthChanged,
}: DeviceCodeAuthPanelProps) {
  const { t } = useTranslation();
  const { message } = useAppMessage();
  const [status, setStatus] = useState<AuthStatusResult>({
    status: provider.auth?.status ?? "not_configured",
    account_label: provider.auth?.account_label ?? "",
    expires_at: provider.auth?.expires_at,
    scopes: provider.auth?.scopes ?? [],
    message: provider.auth?.message ?? "",
  });
  const [flow, setFlow] = useState<AuthStartResult | null>(null);
  const [starting, setStarting] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  const refreshStatus = useCallback(
    async (flowId?: string) => {
      const next = await api.getProviderAuthStatus(provider.id, flowId);
      if (!mounted.current) return next;
      setStatus(next);
      if (next.status === "authenticated") {
        stopPolling();
        setFlow(null);
        onAuthChanged();
      }
      if (next.status === "expired" || next.status === "error") {
        stopPolling();
      }
      return next;
    },
    [onAuthChanged, provider.id, stopPolling],
  );

  const schedulePoll = useCallback(
    (currentFlow: AuthStartResult) => {
      if (!mounted.current) return;
      stopPolling();
      pollTimer.current = setTimeout(async () => {
        try {
          const next = await refreshStatus(currentFlow.flow_id);
          if (mounted.current && next.status === "pending") {
            schedulePoll(currentFlow);
          }
        } catch {
          if (mounted.current) {
            schedulePoll(currentFlow);
          }
        }
      }, secondsToMs(currentFlow.interval));
    },
    [refreshStatus, stopPolling],
  );

  useEffect(() => {
    mounted.current = true;
    void refreshStatus();
    return () => {
      mounted.current = false;
      stopPolling();
    };
  }, [refreshStatus, stopPolling]);

  const handleSignIn = async () => {
    setStarting(true);
    try {
      const start = await api.startProviderAuth(provider.id);
      setFlow(start);
      setStatus({
        status: "pending",
        message: start.message || t("models.providerAuthWaiting"),
      });
      schedulePoll(start);
    } catch (error) {
      const errMsg =
        error instanceof Error ? error.message : t("models.providerAuthFailed");
      message.error(errMsg);
    } finally {
      setStarting(false);
    }
  };

  const handleSignOut = () => {
    Modal.confirm({
      title: t("models.providerAuthSignOut"),
      content: t("models.providerAuthSignOutConfirm", {
        name: provider.name,
      }),
      okText: t("models.providerAuthSignOut"),
      okButtonProps: { danger: true },
      cancelText: t("models.cancel"),
      onOk: async () => {
        setSigningOut(true);
        try {
          const next = await api.logoutProviderAuth(provider.id);
          stopPolling();
          setFlow(null);
          setStatus(next);
          onAuthChanged();
        } catch (error) {
          const errMsg =
            error instanceof Error
              ? error.message
              : t("models.failedToRevoke");
          message.error(errMsg);
        } finally {
          setSigningOut(false);
        }
      },
    });
  };

  const handleCopyCode = async () => {
    if (!flow?.user_code) return;
    try {
      await navigator.clipboard.writeText(flow.user_code);
      message.success(t("models.providerAuthCodeCopied"));
    } catch {
      message.error(t("common.copyFailed"));
    }
  };

  const hint =
    provider.id === "github-copilot"
      ? t("models.providerAuthHintGithubCopilot")
      : typeof provider.meta?.auth_hint === "string"
      ? provider.meta.auth_hint
      : t("models.providerAuthDeviceCodeHint");
  const isAuthenticated = status.status === "authenticated";
  const isWaitingForAuth = status.status === "pending" && Boolean(flow);
  const accountLabel = status.account_label || "";
  const inactiveStatusText =
    status.status === "expired"
      ? t("models.providerAuthExpired")
      : status.status === "error"
      ? status.message || t("models.providerAuthFailed")
      : t("models.providerAuthNotSignedIn");

  return (
    <div
      style={{
        border: "1px solid rgba(0,0,0,0.08)",
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
        background: "rgba(0,0,0,0.02)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>
        {t("models.providerAuthTitle", { name: provider.name })}
      </div>
      <div
        style={{
          color: "rgba(0,0,0,0.65)",
          fontSize: 13,
          lineHeight: 1.6,
          marginBottom: 12,
        }}
      >
        {hint}
      </div>
      <div style={{ marginBottom: 12 }}>
        <span style={{ color: "rgba(0,0,0,0.55)", marginRight: 8 }}>
          {t("models.providerAuthStatus")}:
        </span>
        {isAuthenticated ? (
          <span style={{ color: "#16a34a", display: "inline-flex", gap: 6 }}>
            <CheckCircleFilled />
            {accountLabel
              ? t("models.providerAuthSignedInAs", { account: accountLabel })
              : t("models.providerAuthSignedIn")}
          </span>
        ) : status.status === "pending" ? (
          <span style={{ color: "#d97706", display: "inline-flex", gap: 6 }}>
            <LoadingOutlined />
            {t("models.providerAuthWaiting")}
          </span>
        ) : (
          <span style={{ color: "rgba(0,0,0,0.55)" }}>
            {inactiveStatusText}
          </span>
        )}
      </div>
      {flow && status.status === "pending" && (
        <div
          style={{
            border: "1px dashed rgba(0,0,0,0.18)",
            borderRadius: 6,
            background: "#fff",
            padding: 12,
            marginBottom: 12,
          }}
        >
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ color: "rgba(0,0,0,0.55)", minWidth: 110 }}>
              {t("models.providerAuthUserCode")}
            </span>
            <code style={{ fontSize: 18, fontWeight: 600 }}>
              {flow.user_code}
            </code>
            <Button size="small" icon={<CopyOutlined />} onClick={handleCopyCode}>
              {t("common.copy")}
            </Button>
          </div>
          <div
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              marginTop: 10,
            }}
          >
            <span style={{ color: "rgba(0,0,0,0.55)", minWidth: 110 }}>
              {t("models.providerAuthVerificationUrl")}
            </span>
            <a
              href={flow.verification_uri ?? undefined}
              target="_blank"
              rel="noreferrer"
              style={{ wordBreak: "break-all" }}
            >
              {flow.verification_uri}
            </a>
            {flow.verification_uri && (
              <Button
                size="small"
                icon={<ExportOutlined />}
                onClick={() => {
                  window.open(flow.verification_uri ?? "", "_blank", "noopener");
                }}
              >
                {t("models.providerAuthOpen")}
              </Button>
            )}
          </div>
          <div
            style={{
              color: "rgba(0,0,0,0.55)",
              fontSize: 12,
              lineHeight: 1.5,
              marginTop: 10,
            }}
          >
            {t("models.providerAuthDeviceCodeInstructions")}
          </div>
        </div>
      )}
      <div style={{ display: "flex", gap: 8 }}>
        {isAuthenticated ? (
          <Button
            icon={<LogoutOutlined />}
            loading={signingOut}
            onClick={handleSignOut}
          >
            {t("models.providerAuthSignOut")}
          </Button>
        ) : (
          <Button
            type="primary"
            loading={starting}
            disabled={isWaitingForAuth}
            onClick={handleSignIn}
          >
            {isWaitingForAuth
              ? t("models.providerAuthWaiting")
              : t("models.providerAuthSignIn")}
          </Button>
        )}
      </div>
    </div>
  );
}
