import type { ProviderInfo } from "../../../../../api/types";
import { DeviceCodeAuthPanel } from "./DeviceCodeAuthPanel";

interface ProviderAuthPanelProps {
  provider: ProviderInfo;
  onAuthChanged: () => void;
}

export function ProviderAuthPanel({
  provider,
  onAuthChanged,
}: ProviderAuthPanelProps) {
  if (provider.auth_type === "oauth_device_code") {
    return (
      <DeviceCodeAuthPanel provider={provider} onAuthChanged={onAuthChanged} />
    );
  }
  return null;
}
