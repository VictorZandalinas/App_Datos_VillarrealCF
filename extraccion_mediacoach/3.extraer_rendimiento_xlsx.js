// 3.extraer_rendimiento_xlsx.js (Modificado para aislar los datos de 'Físico')

const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');
const parquet = require('parquetjs');

// Configuración
const BASE_PATH = 'VCF_Mediacoach_Data';
const OUTPUT_BASE_PATH = 'data';

// --- CAMBIO CLAVE ---
// Los datos de 'Físico' se desvían a un archivo PROVISIONAL.
const SHEET_TO_FILE_MAP = {
    'Físico': 'rendimiento_fisico_provisional.parquet',
    'Fisico': 'rendimiento_fisico_provisional.parquet',
    '5':      'rendimiento_5_provisional.parquet', // <-- Correcto
    '10':     'rendimiento_10_provisional.parquet', // <-- Correcto
    '15':     'rendimiento_15_provisional.parquet'  // <-- Correcto
};

const TRACKING_FILE = path.join(OUTPUT_BASE_PATH, 'archivos_procesados.json');

// --- El resto del script es idéntico al que ya funcionaba bien ---

function cargarArchivosProcesados() {
    try {
        if (fs.existsSync(TRACKING_FILE)) { return JSON.parse(fs.readFileSync(TRACKING_FILE, 'utf8')); }
        return {};
    } catch (error) { console.log(`⚠️ Error leyendo historial: ${error.message}`); return {}; }
}
function guardarArchivosProcesados(tracking) {
    try { fs.writeFileSync(TRACKING_FILE, JSON.stringify(tracking, null, 2)); }
    catch (error) { console.log(`❌ Error guardando historial: ${error.message}`); }
}
function generarClaveArchivo(rutaArchivo) {
    const stats = fs.statSync(rutaArchivo);
    return `${path.basename(rutaArchivo)}_${stats.size}_${stats.mtime.getTime()}`;
}
function archivoYaProcesado(rutaArchivo, tracking) { return tracking.hasOwnProperty(generarClaveArchivo(rutaArchivo)); }
function marcarArchivoProcesado(rutaArchivo, tracking) {
    tracking[generarClaveArchivo(rutaArchivo)] = { archivo: path.basename(rutaArchivo), procesado: new Date().toISOString() };
}
function obtenerNombreArchivoParquet(nombreHoja) {
    const nombreNormalizado = nombreHoja.trim();
    if (SHEET_TO_FILE_MAP[nombreNormalizado]) return SHEET_TO_FILE_MAP[nombreNormalizado];
    const nombreLower = nombreNormalizado.toLowerCase();
    for (const [sheetName, fileName] of Object.entries(SHEET_TO_FILE_MAP)) { if (sheetName.toLowerCase() === nombreLower) return fileName; }
    return `rendimiento_${nombreNormalizado.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}.parquet`;
}
async function leerDatosParquetExistentes(rutaArchivo) {
    try {
        if (!fs.existsSync(rutaArchivo)) return [];
        const reader = await parquet.ParquetReader.openFile(rutaArchivo);
        const cursor = reader.getCursor();
        const datos = [];
        let record = null;
        while (record = await cursor.next()) { datos.push(record); }
        await reader.close();
        return datos;
    } catch (error) {
        if (fs.existsSync(rutaArchivo)) {
            const backupPath = rutaArchivo + '.backup_' + Date.now();
            fs.copyFileSync(rutaArchivo, backupPath);
            console.log(`💾 Backup de archivo corrupto creado: ${path.basename(backupPath)}`);
        }
        return null;
    }
}
async function escribirParquet(datos, rutaCompleta) {
    if (datos.length === 0) { console.log(`⚠️ No hay datos para escribir en ${path.basename(rutaCompleta)}`); return true; }
    try {
        const schemaDef = {};
        const primerFila = datos[0];
        for (const campo of Object.keys(primerFila)) {
            const valor = primerFila[campo];
            if (typeof valor === 'number') { schemaDef[campo] = { type: Number.isInteger(valor) ? 'INT64' : 'DOUBLE', optional: true }; }
            else { schemaDef[campo] = { type: 'UTF8', optional: true }; }
        }
        const schema = new parquet.ParquetSchema(schemaDef);
        const writer = await parquet.ParquetWriter.openFile(schema, rutaCompleta);
        for (const fila of datos) { await writer.appendRow(fila); }
        await writer.close();
        console.log(`✅ Escritura exitosa para ${path.basename(rutaCompleta)} (${datos.length} filas).`);
        return true;
    } catch (error) { console.log(`❌ Error escribiendo en ${path.basename(rutaCompleta)}: ${error.message}`); return false; }
}
function generarClaveUnicaMejorada(fila) {
    const esHojaIntervalos = fila['Inicio intervalo'] !== undefined;
    let campos;
    if (esHojaIntervalos) { campos = [fila['Id Jugador'] || '', fila.Partido || '', fila.Periodo || '', fila['Inicio intervalo'] || '', fila['Fin intervalo'] || '', fila.archivo_origen || '', fila.tipo_reporte || '']; }
    else { campos = [fila['Id Jugador'] || '', fila.Partido || '', fila.Equipo || '', fila.archivo_origen || '', fila.tipo_reporte || '', fila.hoja || '']; }
    return campos.join('|').toLowerCase().trim();
}
function filtrarDatosDuplicados(datosNuevos, clavesExistentes) {
    const datosFiltrados = [];
    datosNuevos.forEach((fila) => {
        if (!clavesExistentes.has(generarClaveUnicaMejorada(fila))) { datosFiltrados.push(fila); }
    });
    return datosFiltrados;
}

function extraerMetadatosDeEncabezado(sheet) {
    const range = XLSX.utils.decode_range(sheet['!ref']);
    for (let row = 0; row <= Math.min(10, range.e.r); row++) {
        for (let col = 0; col <= Math.min(10, range.e.c); col++) {
            const cell = sheet[XLSX.utils.encode_cell({ r: row, c: col })];
            if (cell && typeof cell.v === 'string') {
                const valorOriginal = cell.v.trim();
                // Patrón para capturar: COMPETICION | Temporada XXXX - XXXX | JX | EQUIPO vs EQUIPO (YYYY-MM-DD)
                const patron = /^(.+?)\s*\|\s*Temporada\s+(\d{4}\s*-\s*\d{4})\s*\|\s*J\d+\s*\|\s*.+?\((\d{4}-\d{2}-\d{2})\)/i;
                const match = valorOriginal.match(patron);
                if (match) {
                    return {
                        competicion: match[1].trim(),
                        temporada: match[2].replace(/\s/g, ''), // Quitar espacios: "2025-2026"
                        fecha: match[3]
                    };
                }
            }
        }
    }
    return null;
}

function extraerJornadaPartido(nombreCarpeta) {
    let jornada = null, partido = null;
    if (nombreCarpeta.toLowerCase().startsWith('j') && nombreCarpeta.length > 1) {
        let finJornada = -1;
        for (let i = 1; i < nombreCarpeta.length; i++) { if (!nombreCarpeta[i].match(/\d/)) { finJornada = i; break; } }
        jornada = nombreCarpeta.substring(0, finJornada !== -1 ? finJornada : nombreCarpeta.length);
    }
    if (nombreCarpeta.includes('_')) { partido = nombreCarpeta.substring(nombreCarpeta.indexOf('_') + 1); }
    return { jornada, partido };
}
function extraerEquipo(sheet) {
    const range = XLSX.utils.decode_range(sheet['!ref']);
    for (let row = 0; row <= Math.min(50, range.e.r); row++) {
        for (let col = 0; col <= Math.min(30, range.e.c); col++) {
            const cell = sheet[XLSX.utils.encode_cell({ r: row, c: col })];
            if (cell && typeof cell.v === 'string') {
                const valorOriginal = cell.v.trim();
                const patrones = [/informe\s+de\s+rendimiento\s+físico\s+intervalos\s+\d+['´]\s+(.+)/i, /informe\s+de\s+rendimiento\s+físico\s+(?!intervalos)(.+)/i, /informe\s+de\s+rendimiento\s+intervalos\s+\d+['´]\s+(.+)/i, /informe\s+de\s+rendimiento\s+(?!físico|intervalos)(.+)/i, /informe\s+de\s+rendimiento\s+(.+)/i];
                for (const patron of patrones) {
                    const match = valorOriginal.match(patron);
                    if (match && match[1]) {
                        let equipo = match[1].trim().replace(/\s*(vs\.?|contra|v\.?)\s+.*/i, '').replace(/\s*-\s*.*/i, '').replace(/\s*\(.*\)/g, '').replace(/\s*\[.*\]/g, '').replace(/\s+(jornada|partido|fecha|temporada)\s+.*/i, '');
                        if (equipo.length > 2 && equipo.length < 50 && !equipo.match(/^\d+$/)) { return equipo; }
                    }
                }
            }
        }
    }
    return null;
}
function encontrarFilaHeaders(sheet) {
    const range = XLSX.utils.decode_range(sheet['!ref']);
    for (let row = 0; row <= Math.min(20, range.e.r); row++) {
        for (let col = 0; col <= Math.min(10, range.e.c); col++) {
            const cell = sheet[XLSX.utils.encode_cell({ r: row, c: col })];
            if (cell && cell.v && String(cell.v).trim().toLowerCase() === 'id jugador') { return row; }
        }
    }
    return null;
}
function limpiarColumnasMetadatos(headers) {
    const columnasMetadatosConflictivas = ['temporada', 'competicion', 'liga', 'jornada', 'partido', 'equipo', 'season', 'competition', 'matchday', 'match', 'team'];
    return headers.filter(header => {
        if (!header || header.trim() === '') return false;
        return !columnasMetadatosConflictivas.includes(header.toLowerCase().trim());
    });
}
function añadirMetadatosEstandarizados(obj, equipo, jornada, partido, tipoReporte, nombreHoja, nombreArchivo, metadatos) {
    // Solo usar valores extraídos del archivo, sin valores por defecto
    obj.Temporada = metadatos?.temporada || null;
    obj.Competicion = metadatos?.competicion || null;
    obj.Fecha = metadatos?.fecha || null;
    obj.Jornada = jornada;
    obj.Partido = partido;
    obj.Equipo = equipo;
    obj.tipo_reporte = tipoReporte;
    obj.hoja = nombreHoja;
    obj.archivo_origen = nombreArchivo;
    return obj;
}
function procesarHojaXlsx(sheet, nombreHoja, jornada, partido, tipoReporte, nombreArchivo) {
    const equipo = extraerEquipo(sheet);
    if (!equipo) return null;
    const metadatos = extraerMetadatosDeEncabezado(sheet);
    const filaHeaders = encontrarFilaHeaders(sheet);
    if (filaHeaders === null) return null;
    const jsonData = XLSX.utils.sheet_to_json(sheet, { range: filaHeaders });
    if (jsonData.length === 0) return null;
    const headersLimpios = limpiarColumnasMetadatos(Object.keys(jsonData[0]));
    return jsonData.map(rowOriginal => {
        const obj = {};
        headersLimpios.forEach(header => { if (rowOriginal.hasOwnProperty(header)) { obj[header] = rowOriginal[header]; } });
        return añadirMetadatosEstandarizados(obj, equipo, jornada, partido, tipoReporte, nombreHoja, nombreArchivo, metadatos);
    }).filter(row => row['Id Jugador'] != null);
}
function procesarArchivoXlsx(archivoPath, jornada, partido, tipoReporte) {
    const nombreArchivo = path.basename(archivoPath);
    console.log(`  📄 Procesando: ${nombreArchivo}`);
    try {
        const workbook = XLSX.readFile(archivoPath, { cellStyles: false, cellNF: false });
        const resultadosPorHoja = {};
        workbook.SheetNames.forEach(nombreHoja => {
            const nombreHojaNormalizado = nombreHoja.trim().toLowerCase();
            if (nombreHojaNormalizado === 'tiempo extra') return;
            const nombreArchivoParquet = obtenerNombreArchivoParquet(nombreHoja);
            if (nombreArchivoParquet.startsWith('rendimiento_')) {
                const datosHoja = procesarHojaXlsx(workbook.Sheets[nombreHoja], nombreHoja, jornada, partido, tipoReporte, nombreArchivo);
                if (datosHoja && datosHoja.length > 0) {
                    const claveHoja = Object.keys(SHEET_TO_FILE_MAP).find(k => k.toLowerCase() === nombreHojaNormalizado) || nombreHoja;
                    if (!resultadosPorHoja[claveHoja]) { resultadosPorHoja[claveHoja] = []; }
                    resultadosPorHoja[claveHoja].push(...datosHoja);
                }
            }
        });
        return resultadosPorHoja;
    } catch (error) { console.log(`    ❌ Error en ${nombreArchivo}: ${error.message}`); return null; }
}
function buscarArchivosXlsxEnCarpeta(carpetaPath) {
    try {
        const archivos = fs.readdirSync(carpetaPath);
        const rendimiento1 = archivos.find(f => f.toLowerCase().includes('rendimiento_1_') && f.endsWith('.xlsx'));
        const rendimiento2 = archivos.find(f => f.toLowerCase().includes('rendimiento_2_') && f.endsWith('.xlsx'));
        return { rendimiento1: rendimiento1 ? path.join(carpetaPath, rendimiento1) : null, rendimiento2: rendimiento2 ? path.join(carpetaPath, rendimiento2) : null };
    } catch (error) { return { rendimiento1: null, rendimiento2: null }; }
}
function encontrarCarpetasDePartidos(directorio) {
    let carpetasEncontradas = [];
    try {
        const items = fs.readdirSync(directorio, { withFileTypes: true });
        for (const item of items) {
            const rutaCompleta = path.join(directorio, item.name);
            if (item.isDirectory()) {
                if (item.name === 'Partidos') {
                    carpetasEncontradas = carpetasEncontradas.concat(fs.readdirSync(rutaCompleta, { withFileTypes: true }).filter(sub => sub.isDirectory()).map(sub => path.join(rutaCompleta, sub.name)));
                } else {
                    carpetasEncontradas = carpetasEncontradas.concat(encontrarCarpetasDePartidos(rutaCompleta));
                }
            }
        }
    } catch (error) { console.error(`❌ Error al leer ${directorio}: ${error.message}`); }
    return carpetasEncontradas;
}
async function main() {
    console.log('🚀 Iniciando extracción de datos...');
    if (!fs.existsSync(OUTPUT_BASE_PATH)) { fs.mkdirSync(OUTPUT_BASE_PATH); }
    const tracking = cargarArchivosProcesados();
    if (!fs.existsSync(BASE_PATH)) { console.log(`❌ No se encuentra: ${BASE_PATH}`); return; }

    const carpetasDePartidos = encontrarCarpetasDePartidos(BASE_PATH);
    console.log(`📁 Encontradas ${carpetasDePartidos.length} carpetas de partidos.`);
    const datosNuevosPorHoja = {};
    let archivosProcesadosContador = 0;

    for (const carpetaPath of carpetasDePartidos) {
        const nombreCarpeta = path.basename(carpetaPath);
        const archivos = buscarArchivosXlsxEnCarpeta(carpetaPath);
        const { jornada, partido } = extraerJornadaPartido(nombreCarpeta);
        const archivosAProcesar = [{ tipo: 'rendimiento_1', ruta: archivos.rendimiento1 }, { tipo: 'rendimiento_2', ruta: archivos.rendimiento2 }];

        for (const infoArchivo of archivosAProcesar) {
            if (infoArchivo.ruta && !archivoYaProcesado(infoArchivo.ruta, tracking)) {
                const resultados = procesarArchivoXlsx(infoArchivo.ruta, jornada, partido, infoArchivo.tipo);
                if (resultados) {
                    Object.entries(resultados).forEach(([nombreHoja, datos]) => {
                        if (!datosNuevosPorHoja[nombreHoja]) datosNuevosPorHoja[nombreHoja] = [];
                        datosNuevosPorHoja[nombreHoja].push(...datos);
                    });
                    marcarArchivoProcesado(infoArchivo.ruta, tracking);
                    archivosProcesadosContador++;
                }
            }
        }
    }

    guardarArchivosProcesados(tracking);
    console.log(`\n📊 Resumen de lectura: ${archivosProcesadosContador} archivos nuevos procesados.`);
    if (Object.keys(datosNuevosPorHoja).length === 0) { console.log('\n⚠️ No se encontraron datos nuevos. Proceso finalizado.'); return; }

    console.log('\n✍️  Guardando datos extraídos...');
    for (const [nombreHoja, datosNuevos] of Object.entries(datosNuevosPorHoja)) {
        const nombreArchivo = obtenerNombreArchivoParquet(nombreHoja);
        const rutaCompleta = path.join(OUTPUT_BASE_PATH, nombreArchivo);
        
        // Para el archivo PROVISIONAL, siempre sobreescribimos con lo nuevo.
        if (nombreArchivo.includes('_provisional')) {
             console.log(`  Writing provisional data for ${nombreHoja}...`);
             await escribirParquet(datosNuevos, rutaCompleta);
        } else {
            // Para el resto, hacemos la lógica de añadir sin duplicar.
             console.log(`  Merging data for ${nombreHoja}...`);
            const datosExistentes = await leerDatosParquetExistentes(rutaCompleta);
            if (datosExistentes === null) {
                await escribirParquet(datosNuevos, rutaCompleta);
            } else {
                const clavesExistentes = new Set(datosExistentes.map(generarClaveUnicaMejorada));
                const datosFiltrados = filtrarDatosDuplicados(datosNuevos, clavesExistentes);
                if (datosFiltrados.length > 0) {
                    await escribirParquet([...datosExistentes, ...datosFiltrados], rutaCompleta);
                } else {
                    console.log(`  ✅ No hay filas nuevas para ${nombreArchivo}.`);
                }
            }
        }
    }
    console.log('\n🎉 Proceso de extracción completado!');
    console.log('💡 Ahora puedes ejecutar "node 5.fusionar_fisico.js" para unir los datos físicos.');
}
if (require.main === module) { main().catch(console.error); }