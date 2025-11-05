import extractMain from "#repo/services/extract/main.go?raw";
import extractTelemetry from "#repo/services/extract/telemetry.go?raw";

const extractConstant = (source: string, constantName: string) => {
  const regex = new RegExp(`${constantName}\\s+=\\s+"([^"]+)"`);
  const match = source.match(regex);
  return match?.[1] ?? null;
};

const extractDuration = (source: string, constantName: string) => {
  const regex = new RegExp(`${constantName}\\s+=\\s+([^\\n]+)`);
  const match = source.match(regex);
  return match?.[1]?.trim() ?? null;
};

const extractStructFields = (source: string, structName: string) => {
  const regex = new RegExp(`type\\s+${structName}\\s+struct\\s+{([\\s\\S]*?)}`);
  const match = source.match(regex);
  if (!match) return [] as string[];
  return match[1]
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("//"));
};

export const telemetryDefaults = {
  library: extractConstant(extractMain, "defaultTelemetryLibrary"),
  operation: extractConstant(extractMain, "defaultTelemetryOperation"),
  httpTimeout: extractDuration(extractMain, "defaultHTTPClientTimeout"),
  dialTimeout: extractDuration(extractMain, "defaultDialTimeout"),
  callTimeout: extractDuration(extractMain, "defaultCallTimeout")
};

export const telemetryRecordFields = extractStructFields(
  extractTelemetry,
  "telemetryRecord"
);

export const telemetryConfigFields = extractStructFields(
  extractTelemetry,
  "telemetryConfig"
);
