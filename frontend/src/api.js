export async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error(
      "Serveur inaccessible. Vérifiez que le serveur Python est lancé.",
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `La requête a échoué (${response.status}).`,
    );
  }
  return response;
}

export async function json(path, options) {
  return (await request(path, options)).json();
}

export function uploadImage(file, iou, signal) {
  const body = new FormData();
  body.append("file", file);
  return json(`/api/image?iou=${iou}`, { method: "POST", body, signal });
}
