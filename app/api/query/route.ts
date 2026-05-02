import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const backendBaseUrl = () => process.env.NEXT_PUBLIC_API_URL?.trim();

export async function POST(request: NextRequest) {
  const apiBaseUrl = backendBaseUrl();
  if (!apiBaseUrl) {
    return Response.json(
      { message: "NEXT_PUBLIC_API_URL is not configured." },
      { status: 500 },
    );
  }

  const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Forwarded-Host": request.headers.get("host") ?? "",
    },
    body: await request.text(),
    cache: "no-store",
  });

  const headers = new Headers(response.headers);
  headers.set("Cache-Control", "no-cache");

  return new Response(response.body, {
    status: response.status,
    headers,
  });
}
