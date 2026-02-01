import { tool } from "@opencode-ai/plugin";

const blockedHosts = new Map<string, number>(); // host -> times blocked this run/process

function uniqKeepOrder(items: string[]) {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const x of items) {
    const k = x.trim();
    if (!k) continue;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  return out;
}

function extractUrls(text: string) {
  const re = /\bhttps?:\/\/[^\s<>()\]]+/g;
  const matches = text.match(re) ?? [];
  return uniqKeepOrder(matches.map(u => u.replace(/[),.;\]]+$/g, "")));
}

function getHost(rawUrl: string) {
  try {
    return new URL(rawUrl).hostname.toLowerCase();
  } catch {
    return "";
  }
}

function detectHumanVerification(md: string) {
  const low = md.toLowerCase();
  const markers = [
    "verifying you are human",
    "verification required",
    "checking your browser",
    "just a moment",
    "enable javascript and cookies",
    "cloudflare",
    "captcha",
    "/cdn-cgi/",
    "cf-chl",
    "challenge-platform",
  ];

  const hits = markers.filter(m => low.includes(m));
  const blocked = hits.length > 0;

  return {
    blocked,
    reason: blocked ? "human_verification_interstitial" : "",
    hits,
  };
}

function scoreUrl(u: string) {
  const s = u.toLowerCase();

  const hostBoost =
    (s.includes("figshare.com") ? 50 : 0) +
    (s.includes("zenodo.org") ? 50 : 0) +
    (s.includes("osf.io") ? 35 : 0) +
    (s.includes("dataverse") ? 30 : 0) +
    (s.includes("kaggle.com") ? 25 : 0) +
    (s.includes("github.com") ? 20 : 0) +
    (s.includes("huggingface.co") ? 20 : 0) +
    (s.includes("drive.google.com") ? 15 : 0) +
    (s.includes("dropbox.com") ? 15 : 0);
    
  const intentBoost =
    (s.includes("supplement") || s.includes("supporting") || s.includes("si") ? 18 : 0) +
    (s.includes("download") ? 16 : 0) +
    (s.includes("dataset") || s.includes("data-set") ? 14 : 0) +
    (s.includes("files") ? 8 : 0) +
    (s.includes(".zip") || s.includes(".tar") || s.includes(".gz") || s.includes(".7z") ? 12 : 0) +
    (s.includes(".csv") || s.includes(".tsv") || s.includes(".json") || s.includes(".sdf") || s.includes(".mol") ? 10 : 0) +
    (s.includes(".xlsx") ? 6 : 0) +
    (s.includes(".pdf") ? 2 : 0);

  const penalty =
    (s.includes("twitter.com") || s.includes("x.com") ? 40 : 0) +
    (s.includes("facebook.com") ? 40 : 0) +
    (s.includes("linkedin.com") ? 30 : 0) +
    (s.includes("mailto:") ? 100 : 0) +
    (s.includes("javascript:") ? 100 : 0);

  return hostBoost + intentBoost - penalty;
}

function evidenceSnippets(md: string) {
  const needles = [
    "data availability",
    "availability of data",
    "code availability",
    "supplementary",
    "supporting information",
    "supplemental",
    "dataset",
    "download",
    "figshare",
    "zenodo",
    "osf",
    "dataverse",
    "github",
  ];

  const lines = md.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const low = line.toLowerCase();
    if (needles.some(n => low.includes(n))) {
      const prev = lines[i - 1] ?? "";
      const next = lines[i + 1] ?? "";
      const block = [prev, line, next].filter(Boolean).join("\n");
      out.push(block.trim());
      if (out.length >= 12) break;
    }
  }
  return uniqKeepOrder(out);
}

function truncate(s: string, n: number) {
  if (s.length <= n) return s;
  return s.slice(0, n) + `\n\n---\n[truncated] Returned first ${n} chars of markdown.`;
}

export default tool({
  description:
    "Fetch a URL using Crawl4AI (JS-rendered), then return ranked candidate dataset links + evidence snippets + truncated markdown.",

  args: {
    url: tool.schema.string().describe("URL to crawl"),
    maxChars: tool.schema.number().optional().describe("Max characters of markdown to include (default 20000)"),
    maxLinks: tool.schema.number().optional().describe("Max candidate links to return (default 40)"),
  },

  async execute({ url, maxChars, maxLinks }) {
    const limit = maxChars ?? 20000;
    const linkLimit = maxLinks ?? 40;

    const host = getHost(url);
    const prevBlocks = host ? (blockedHosts.get(host) ?? 0) : 0;

    // HARD GUARD: don't keep retrying a host that already blocked us
    if (host && prevBlocks >= 1) {
      return [
        `# CrawlFetch result`,
        `URL: ${url}`,
        `HOST: ${host}`,
        `BLOCKED_BY_HUMAN_VERIFICATION: true`,
        `BLOCK_REASON: previously_blocked_host`,
        ``,
        `## Page markdown (truncated)`,
        truncate(`Skipped crawling because this host previously returned a human-verification interstitial in this run.`, 2000),
      ].join("\n");
    }

    let mdFull = "";
    try {
      mdFull = await Bun.$`crwl crawl ${url} -o markdown`.text();
    } catch (err: any) {
      const msg = (err && (err.message || String(err))) || "Unknown error";
      return [
        `# CrawlFetch result`,
        `URL: ${url}`,
        host ? `HOST: ${host}` : ``,
        `ERROR: true`,
        `ERROR_MESSAGE: ${msg}`,
      ].filter(Boolean).join("\n");
    }

    // Detect Cloudflare/CAPTCHA/etc.
    const block = detectHumanVerification(mdFull);
    if (block.blocked) {
      if (host) blockedHosts.set(host, prevBlocks + 1);

      return [
        `# CrawlFetch result`,
        `URL: ${url}`,
        host ? `HOST: ${host}` : ``,
        `BLOCKED_BY_HUMAN_VERIFICATION: true`,
        `BLOCK_REASON: ${block.reason}`,
        `BLOCK_HITS: ${block.hits.join(" | ")}`,
        ``,
        `## Page markdown (truncated)`,
        truncate(mdFull, Math.min(limit, 6000)),
      ].filter(Boolean).join("\n");
    }

    // Normal behavior (your original logic)
    const urls = extractUrls(mdFull);
    const ranked = urls
      .map(u => ({ u, score: scoreUrl(u) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, linkLimit);

    const evid = evidenceSnippets(mdFull);
    const md = truncate(mdFull, limit);

    const topLinksText = ranked.length
      ? ranked.map(x => `- (${x.score}) ${x.u}`).join("\n")
      : "(none found)";

    const evidText = evid.length
      ? evid.map((b, i) => `### Evidence ${i + 1}\n${b}`).join("\n\n")
      : "(none found)";

    return [
      `# CrawlFetch result`,
      `URL: ${url}`,
      ``,
      `## Top candidate links`,
      topLinksText,
      ``,
      `## Evidence snippets`,
      evidText,
      ``,
      `## Page markdown (truncated)`,
      md,
    ].join("\n");
  },
});
