"""Herramientas para transcribir archivos MP3 a texto usando OpenAI Whisper.

Requiere instalar previamente el paquete ``openai-whisper`` y ``torch`` compatibles
con la plataforma.
"""

from pathlib import Path
from typing import Optional, Union

import whisper


def transcribir_mp3_a_txt(
    ruta_mp3: Union[str, Path],
    ruta_txt: Optional[Union[str, Path]] = None,
    modelo: str = "base",
    *,
    lenguaje: Optional[str] = "es",
    **opciones_transcripcion,
) -> Path:
    """Transcribe un archivo MP3 a texto plano.

    Parameters
    ----------
    ruta_mp3:
        Ruta del archivo de audio en formato MP3.
    ruta_txt:
        Ruta donde se guardará el resultado en texto plano. Si se omite, se
        utilizará la misma ruta del MP3 cambiando la extensión por ``.txt``.
    modelo:
        Nombre del modelo Whisper a utilizar (por ejemplo: ``tiny``, ``base``,
        ``small``, ``medium`` o ``large``).
    lenguaje:
        Código ISO 639-1 del idioma del audio. Si es ``None`` Whisper intentará
        detectar el idioma automáticamente.
    **opciones_transcripcion:
        Parámetros adicionales que se pasan directamente a ``model.transcribe``.

    Returns
    -------
    Path
        Ruta absoluta del archivo de texto generado.
    """

    ruta_mp3 = Path(ruta_mp3)
    if not ruta_mp3.is_file():
        raise FileNotFoundError(f"No se encontró el archivo MP3: {ruta_mp3}")

    if ruta_txt is None:
        ruta_txt = ruta_mp3.with_suffix(".txt")
    else:
        ruta_txt = Path(ruta_txt)

    ruta_txt.parent.mkdir(parents=True, exist_ok=True)

    # Carga del modelo Whisper
    modelo_whisper = whisper.load_model(modelo)

    # Transcripción del audio
    resultado = modelo_whisper.transcribe(
        str(ruta_mp3), language=lenguaje, **opciones_transcripcion
    )

    texto = resultado.get("text", "").strip()

    if not texto:
        raise ValueError("La transcripción resultó vacía.")

    ruta_txt.write_text(texto, encoding="utf-8")
    return ruta_txt.resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe un archivo MP3 a texto usando OpenAI Whisper.",
    )
    parser.add_argument("audio", type=Path, help="Ruta del archivo MP3")
    parser.add_argument(
        "--salida",
        type=Path,
        help="Ruta del archivo de salida .txt. Por defecto mismo nombre que el MP3.",
    )
    parser.add_argument(
        "--modelo",
        default="base",
        help="Modelo Whisper a utilizar (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--lenguaje",
        default="es",
        help="Código ISO 639-1 del idioma del audio. Use None para autodetección.",
    )

    args = parser.parse_args()

    lenguaje = None if args.lenguaje.lower() == "none" else args.lenguaje

    ruta_generada = transcribir_mp3_a_txt(
        args.audio,
        ruta_txt=args.salida,
        modelo=args.modelo,
        lenguaje=lenguaje,
    )
    print(f"Transcripción guardada en: {ruta_generada}")
