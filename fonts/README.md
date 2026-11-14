# Default font

The source archive does not bundle font binaries. First rendering downloads Noto
Sans CJK SC Regular (Sans2.004) from the official Noto CJK repository and verifies
SHA256. See [OFL.txt](OFL.txt) for the SIL Open Font License. Copyright notices are
also embedded in the original font metadata.

For offline use download
https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
to `fonts/NotoSansCJKsc-Regular.otf` before running.

SHA256: `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`.

Default rendering also downloads Noto Sans Arabic as a fallback. Its license is
in [OFL-Arabic.txt](OFL-Arabic.txt). For offline use download
https://raw.githubusercontent.com/google/fonts/29b4ec6df2522a8aee279414b78a07910b065874/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf
to `fonts/NotoSansArabic-Regular.ttf`.
SHA256: `63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e`.

Alternatively use `--font-path` with your own font; this does not require a network
download. Defaults cover CJK, Latin and Arabic; other scripts may require a suitable
custom font. Default font replacement can change line wrapping and appearance.
