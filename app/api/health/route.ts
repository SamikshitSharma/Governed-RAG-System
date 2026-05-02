export const dynamic = "force-dynamic";

export async function GET() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!apiBaseUrl) {
    return Response.json(
      { status: "misconfigured", detail: "NEXT_PUBLIC_API_URL is not configured." },
      { status: 500 },
    );
  }

  try {
    const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/health`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return Response.json(
        { status: "unhealthy", detail: `Backend returned ${response.status}.` },
        { status: response.status },
      );
    }

    return new Response(await response.text(), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
      },
    });
  } catch (error) {
    return Response.json(
      {
        status: "unreachable",
        detail: error instanceof Error ? error.message : "Backend health check failed.",
      },
      { status: 502 },
    );
  }
}
