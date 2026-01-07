const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');
const parquet = require('parquetjs');

// Configuración
const BASE_DATA_PATH = 'VCF_Mediacoach_Data';
const OUTPUT_BASE_PATH = 'data';

async function leerDatosParquetExistentes(rutaArchivo) {
    try {
        if (fs.existsSync(rutaArchivo)) {
            console.log('📖 Cargando datos existentes del parquet...');
            const reader = await parquet.ParquetReader.openFile(rutaArchivo);
            const cursor = reader.getCursor();
            const datos = [];
            
            let record = null;
            while (record = await cursor.next()) {
                datos.push(record);
            }
            
            await reader.close();
            console.log(`✅ Cargadas ${datos.length} filas existentes`);
            return datos;
        } else {
            console.log('🆕 No existe archivo parquet previo, creando nuevo...');
            return [];
        }
    } catch (error) {
        console.log(`⚠️ Error leyendo parquet existente: ${error.message}`);
        console.log('🆕 Iniciando con datos vacíos...');
        return [];
    }
}

function generarClaveUnica(fila) {
    const campos = [
        fila['Id Jugador'] || fila['id_jugador'] || '',
        fila.Temporada || fila.temporada || '',     // Usar campo estándar
        fila.Competicion || fila.liga || '',        // Usar campo estándar  
        fila.Jornada || fila.jornada || '',
        fila.Partido || fila.partido || '',
        fila.Equipo || fila.equipo || '',
        fila.archivo_origen || '',
        fila.tipo_reporte || ''
    ];
    
    return campos.join('|').toLowerCase().trim();
}

function crearConjuntoDuplicados(datosExistentes) {
    const clavesExistentes = new Set();
    
    datosExistentes.forEach(fila => {
        const clave = generarClaveUnica(fila);
        if (clave && clave !== '|||||||') {
            clavesExistentes.add(clave);
        }
    });
    
    console.log(`🔍 Creado índice de ${clavesExistentes.size} registros únicos existentes`);
    return clavesExistentes;
}

function filtrarDatosDuplicados(datosNuevos, clavesExistentes) {
    const datosFiltrados = [];
    const duplicadosEncontrados = [];
    const duplicadosInternos = new Set();
    
    datosNuevos.forEach((fila, index) => {
        const clave = generarClaveUnica(fila);
        
        if (!clave || clave === '|||||||') {
            console.log(`⚠️ Fila ${index + 1} tiene datos incompletos, se omite`);
            return;
        }
        
        if (clavesExistentes.has(clave)) {
            duplicadosEncontrados.push({
                fila: index + 1,
                jugador: fila['Id Jugador'] || 'Sin ID',
                partido: `${fila.jornada} | ${fila.partido}`,
                equipo: fila.equipo,
                tipo: fila.tipo_reporte
            });
            return;
        }
        
        if (duplicadosInternos.has(clave)) {
            console.log(`🔄 Duplicado interno detectado en fila ${index + 1}`);
            return;
        }
        
        duplicadosInternos.add(clave);
        datosFiltrados.push(fila);
    });
    
    if (duplicadosEncontrados.length > 0) {
        console.log(`\n🚫 Duplicados detectados (NO se añadirán): ${duplicadosEncontrados.length}`);
    }
    
    return {
        datosFiltrados,
        totalDuplicados: duplicadosEncontrados.length,
        duplicadosInternos: datosNuevos.length - datosFiltrados.length - duplicadosEncontrados.length
    };
}

function extraerJornadaPartido(nombreCarpeta) {
    let jornada = null;
    let partido = null;
    
    console.log(`🔍 Analizando carpeta: ${nombreCarpeta}`);
    
    // Caso 1: Carpeta formato "j12_equipo1_vs_equipo2"
    if (nombreCarpeta.toLowerCase().startsWith('j') && nombreCarpeta.length > 1) {
        // Encontrar donde termina la jornada (j + números)
        let i = 1;
        while (i < nombreCarpeta.length && nombreCarpeta[i].match(/\d/)) {
            i++;
        }
        
        jornada = nombreCarpeta.substring(0, i);
        
        // El resto es el partido (después del primer _ si existe)
        let resto = nombreCarpeta.substring(i);
        if (resto.startsWith('_')) {
            partido = resto.substring(1);
        } else {
            partido = resto || nombreCarpeta;
        }
    }
    // Caso 2: Carpeta formato "Partido_12345" 
    else if (nombreCarpeta.toLowerCase().startsWith('partido_')) {
        jornada = 'desconocida'; // No podemos extraer jornada de este formato
        partido = nombreCarpeta;
    }
    // Caso 3: Otros formatos
    else {
        jornada = 'desconocida';
        partido = nombreCarpeta;
    }
    
    console.log(`✅ Extraído - Jornada: '${jornada}', Partido: '${partido}'`);
    return { jornada, partido };
}

function extraerEquipo(sheet) {
    const range = XLSX.utils.decode_range(sheet['!ref']);

    for (let row = 0; row <= Math.min(30, range.e.r); row++) {
        for (let col = 0; col <= Math.min(20, range.e.c); col++) {
            const cellAddress = XLSX.utils.encode_cell({ r: row, c: col });
            const cell = sheet[cellAddress];

            if (cell && typeof cell.v === 'string') {
                const valor = cell.v.trim().toLowerCase();

                if (valor.includes('escenarios de máxima')) {
                    const match = cell.v.match(/Escenarios de Máxima\s+Exigencia\s+(.*)/i);
                    if (match && match[1]) {
                        const equipo = match[1].trim();
                        if (equipo.length > 0) {
                            return equipo;
                        }
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
            const cellAddress = XLSX.utils.encode_cell({r: row, c: col});
            const cell = sheet[cellAddress];
            
            if (cell && cell.v && String(cell.v).trim() === 'Id Jugador') {
                return row;
            }
        }
    }
    return null;
}

function procesarArchivoXlsx(archivoPath, jornada, partido, tipoReporte, temporada, liga) {
    const nombreArchivo = path.basename(archivoPath);
    console.log(`  📄 Procesando: ${nombreArchivo} (${tipoReporte})`);
    
    try {
        const workbook = XLSX.readFile(archivoPath, {
            cellStyles: false,
            cellFormulas: true,   
            cellDates: true,     
            cellNF: false,
            sheetStubs: true     
        });
        
        const sheetName = workbook.SheetNames[0];
        const sheet = workbook.Sheets[sheetName];
        
        const equipo = extraerEquipo(sheet);
        if (!equipo) {
            console.log(`    ❌ No se encontró el equipo`);
            return null;
        }

        const filaHeaders = encontrarFilaHeaders(sheet);
        if (filaHeaders === null) {
            console.log(`    ❌ No se encontró fila con 'Id Jugador'`);
            return null;
        }
        
        console.log(`    📍 Headers en fila: ${filaHeaders + 1}`);
        console.log(`    🏃 Equipo: ${equipo}`);
        
        const jsonData = XLSX.utils.sheet_to_json(sheet, {
            range: filaHeaders,
            header: 1
        });
        
        if (jsonData.length === 0) {
            console.log(`    ❌ No hay datos`);
            return null;
        }
        
        const headers = jsonData[0];
        const headersFiltered = headers.slice(1); // Desde columna 2
        
        const datosProcessed = jsonData.slice(1).map(row => {
            const rowFiltered = row.slice(1); // Desde columna 2
            const obj = {};
            
            headersFiltered.forEach((header, index) => {
                obj[header || `columna_${index + 2}`] = rowFiltered[index];
            });
            
            obj.equipo = equipo;
            obj.liga = liga;           // En lugar de hardcoded 'La Liga'
            obj.temporada = temporada; 
            obj.jornada = jornada;
            obj.partido = partido;
            obj.tipo_reporte = tipoReporte;
            obj.archivo_origen = nombreArchivo;
            
            return obj;
        }).filter(row => {
            return Object.values(row).some(val => val !== null && val !== undefined && val !== '');
        });
        
        console.log(`    ✅ Procesadas ${datosProcessed.length} filas`);
        return datosProcessed;
        
    } catch (error) {
        console.log(`    ❌ Error: ${error.message}`);
        return null;
    }
}

function buscarArchivosXlsxEnCarpeta(carpetaPath) {
    try {
        const archivos = fs.readdirSync(carpetaPath);
        
        // Buscar archivos maxima_exigencia (más flexible)
        let maxima1 = archivos.find(archivo => {
            const archivoLower = archivo.toLowerCase();
            return archivoLower.includes('maxima_exigencia') && 
                   (archivoLower.includes('_1') || archivoLower.includes('1_')) &&
                   archivo.endsWith('.xlsx');
        });
        
        let maxima2 = archivos.find(archivo => {
            const archivoLower = archivo.toLowerCase();
            return archivoLower.includes('maxima_exigencia') && 
                   (archivoLower.includes('_2') || archivoLower.includes('2_')) &&
                   archivo.endsWith('.xlsx');
        });
        
        let tipoArchivo = 'maxima_exigencia';
        
        // Si no se encuentran, buscar archivos otro_xlsx (como está en el código original)
        if (!maxima1 && !maxima2) {
            console.log(`    🔍 No se encontraron maxima_exigencia, buscando otro_xlsx...`);
            
            const archivosOtro = archivos.filter(archivo => 
                archivo.toLowerCase().startsWith('otro_xlsx') && archivo.endsWith('.xlsx')
            );
            
            if (archivosOtro.length >= 2) {
                maxima1 = path.join(carpetaPath, archivosOtro[0]);
                maxima2 = path.join(carpetaPath, archivosOtro[1]);
                tipoArchivo = 'otro_xlsx';
                console.log(`    ✅ Encontrados archivos otro_xlsx: ${archivosOtro[0]}, ${archivosOtro[1]}`);
            } else if (archivosOtro.length === 1) {
                maxima1 = path.join(carpetaPath, archivosOtro[0]);
                tipoArchivo = 'otro_xlsx';
                console.log(`    ⚠️ Solo encontrado un archivo otro_xlsx: ${archivosOtro[0]}`);
            } else {
                // Último intento: buscar cualquier XLSX que pueda ser de máxima exigencia
                console.log(`    🔍 Buscando cualquier XLSX que pueda contener datos de máxima exigencia...`);
                const archivosXlsx = archivos.filter(archivo => archivo.endsWith('.xlsx'));
                
                if (archivosXlsx.length >= 1) {
                    maxima1 = path.join(carpetaPath, archivosXlsx[0]);
                    if (archivosXlsx.length > 1) {
                        maxima2 = path.join(carpetaPath, archivosXlsx[1]);
                    }
                    tipoArchivo = 'generico';
                    console.log(`    🔄 Usando archivos XLSX genéricos: ${archivosXlsx.slice(0, 2).join(', ')}`);
                }
            }
        } else {
            if (maxima1) maxima1 = path.join(carpetaPath, maxima1);
            if (maxima2) maxima2 = path.join(carpetaPath, maxima2);
        }
        
        return {
            maxima1,
            maxima2,
            tipoArchivo
        };
    } catch (error) {
        console.log(`    ❌ Error leyendo carpeta: ${error.message}`);
        return { maxima1: null, maxima2: null, tipoArchivo: 'desconocido' };
    }
}


async function escribirParquet(datos, nombreArchivo) {
    try {
        if (datos.length === 0) {
            console.log('⚠️ No hay datos para escribir');
            return false;
        }

        const primerFilaDatos = datos[0];
        const schemaDef = {};

        Object.keys(primerFilaDatos).forEach(campo => {
            const valorEjemplo = primerFilaDatos[campo];
            let tipo = { type: 'UTF8' };

            if (typeof valorEjemplo === 'number') {
                if (Number.isInteger(valorEjemplo)) {
                    tipo = { type: 'INT64' };
                } else {
                    tipo = { type: 'DOUBLE' };
                }
            } else if (typeof valorEjemplo === 'boolean') {
                tipo = { type: 'BOOLEAN' };
            }

            schemaDef[campo] = tipo;
        });

        const schema = new parquet.ParquetSchema(schemaDef);
        const writer = await parquet.ParquetWriter.openFile(schema, nombreArchivo);

        for (const fila of datos) {
            await writer.appendRow(fila);
        }

        await writer.close();
        return true;
    } catch (error) {
        console.log(`❌ Error escribiendo parquet: ${error.message}`);
        return false;
    }
}

async function main() {
    console.log('🚀 Iniciando procesamiento incremental de archivos XLSX de máxima exigencia...');
    console.log(`🔍 Buscando en: ${BASE_DATA_PATH}`);
    console.log('='.repeat(70));
    
    // Crear carpeta data si no existe
    if (!fs.existsSync(OUTPUT_BASE_PATH)) {
        fs.mkdirSync(OUTPUT_BASE_PATH);
        console.log('📁 Carpeta data creada');
    }
    
    // Verificar que existe la ruta base
    if (!fs.existsSync(BASE_DATA_PATH)) {
        console.log(`❌ No se encuentra la ruta: ${BASE_DATA_PATH}`);
        return;
    }
    
    // Cargar datos existentes del parquet
    const OUTPUT_PATH = path.join(OUTPUT_BASE_PATH, 'maxima_exigencia.parquet');
    const datosExistentes = await leerDatosParquetExistentes(OUTPUT_PATH);
    const clavesExistentes = crearConjuntoDuplicados(datosExistentes);
    
    console.log('-'.repeat(70));
    
    // Estructuras globales para acumular datos de todas las temporadas/ligas
    const datosNuevosSinFiltrar = [];
    let totalCarpetasProcesadas = 0;
    let totalArchivosExitosos = 0;
    let totalErrores = 0;
    
    // Recorrer todas las temporadas
    for (const temporada of fs.readdirSync(BASE_DATA_PATH)) {
        const temporadaPath = path.join(BASE_DATA_PATH, temporada);
        
        if (!fs.statSync(temporadaPath).isDirectory() || (!temporada.startsWith('Temporada_') && !temporada.startsWith('Season_'))) {
            continue;
        }
        
        console.log(`\n=== PROCESANDO TEMPORADA: ${temporada} ===`);
        
        // Recorrer todas las ligas
        for (const liga of fs.readdirSync(temporadaPath)) {
            const ligaPath = path.join(temporadaPath, liga);
            
            if (!fs.statSync(ligaPath).isDirectory()) continue;
            
            const partidosPath = path.join(ligaPath, 'Partidos');
            
            if (!fs.existsSync(partidosPath)) continue;
            
            console.log(`\n--- Procesando liga: ${liga} ---`);
            
            // Buscar carpetas de partidos en esta liga específica
            const carpetas = fs.readdirSync(partidosPath).filter(item => {
                const carpetaPath = path.join(partidosPath, item);
                if (!fs.statSync(carpetaPath).isDirectory()) return false;
                
                // Buscar carpetas que empiecen por 'j' O 'Partido_'
                const itemLower = item.toLowerCase();
                return (itemLower.startsWith('j') && item.length > 1 && item[1].match(/\d/)) ||
                       itemLower.startsWith('partido_');
            });
            
            console.log(`📁 Carpetas de partidos encontradas: ${carpetas.length}`);
            
            // Contadores para esta liga
            let carpetasProcesadas = 0;
            let archivosExitosos = 0;
            let errores = 0;
            
            for (const carpeta of carpetas) {
                const carpetaPath = path.join(partidosPath, carpeta);
                const { jornada, partido } = extraerJornadaPartido(carpeta);
                
                console.log(`\n📂 Procesando: ${carpeta}`);
                console.log(`  📊 Jornada: ${jornada}, Partido: ${partido}`);
                
                const archivos = buscarArchivosXlsxEnCarpeta(carpetaPath);
                
                if (!archivos.maxima1 && !archivos.maxima2) {
                    console.log(`  ❌ No se encontraron archivos de máxima exigencia ni otro_xlsx`);
                    errores++;
                    continue;
                }
                
                console.log(`  📋 Tipo de archivos encontrados: ${archivos.tipoArchivo}`);
                carpetasProcesadas++;
                let archivosEnEstaCarpeta = 0;
                
                // Determinar nombres de tipos de reporte según el tipo de archivo
                const tipoReporte1 = archivos.tipoArchivo === 'maxima_exigencia' ? 'maxima_exigencia_1' : 'otro_xlsx_1';
                const tipoReporte2 = archivos.tipoArchivo === 'maxima_exigencia' ? 'maxima_exigencia_2' : 'otro_xlsx_2';
                
                // Procesar primer archivo
                if (archivos.maxima1) {
                    console.log(`  🔄 Procesando archivo 1...`);
                    const datos1 = procesarArchivoXlsx(archivos.maxima1, jornada, partido, tipoReporte1, temporada, liga);
                    if (datos1 && datos1.length > 0) {
                        datosNuevosSinFiltrar.push(...datos1);
                        archivosExitosos++;
                        archivosEnEstaCarpeta++;
                        console.log(`    ✅ Archivo 1 procesado exitosamente`);
                    } else {
                        errores++;
                        console.log(`    ❌ Error procesando archivo 1`);
                    }
                } else {
                    console.log(`  ⚠️ Archivo 1 NO encontrado`);
                }
                
                // Procesar segundo archivo
                if (archivos.maxima2) {
                    console.log(`  🔄 Procesando archivo 2...`);
                    const datos2 = procesarArchivoXlsx(archivos.maxima2, jornada, partido, tipoReporte2, temporada, liga);
                    if (datos2 && datos2.length > 0) {
                        datosNuevosSinFiltrar.push(...datos2);
                        archivosExitosos++;
                        archivosEnEstaCarpeta++;
                        console.log(`    ✅ Archivo 2 procesado exitosamente`);
                    } else {
                        errores++;
                        console.log(`    ❌ Error procesando archivo 2`);
                    }
                } else {
                    console.log(`  ⚠️ Archivo 2 NO encontrado`);
                }
                
                console.log(`  📊 Archivos procesados en esta carpeta: ${archivosEnEstaCarpeta}/2`);
                console.log('-'.repeat(50));
            }
            
            // Resumen por liga
            console.log(`\n📊 Resumen ${liga}:`);
            console.log(`  📁 Carpetas procesadas: ${carpetasProcesadas}`);
            console.log(`  ✅ Archivos procesados exitosamente: ${archivosExitosos}`);
            console.log(`  ❌ Archivos con errores: ${errores}`);
            
            // Acumular totales globales
            totalCarpetasProcesadas += carpetasProcesadas;
            totalArchivosExitosos += archivosExitosos;
            totalErrores += errores;
        }
    }
    
    // RESUMEN GLOBAL Y PROCESAMIENTO FINAL
    console.log(`\n🎯 RESUMEN GLOBAL DEL PROCESAMIENTO:`);
    console.log(`  📁 Total carpetas procesadas: ${totalCarpetasProcesadas}`);
    console.log(`  ✅ Total archivos procesados exitosamente: ${totalArchivosExitosos}`);
    console.log(`  ❌ Total archivos con errores: ${totalErrores}`);
    console.log(`  📥 Filas candidatas obtenidas: ${datosNuevosSinFiltrar.length}`);
    console.log(`  📚 Filas existentes en parquet: ${datosExistentes.length}`);
    
    if (datosNuevosSinFiltrar.length === 0) {
        console.log('\n⚠️ No se obtuvieron datos nuevos. No se actualiza el parquet.');
        return;
    }
    
    // Filtrar duplicados
    console.log('\n🔍 Filtrando duplicados...');
    const resultadoFiltrado = filtrarDatosDuplicados(datosNuevosSinFiltrar, clavesExistentes);
    const datosNuevosUnicos = resultadoFiltrado.datosFiltrados;
    
    console.log(`\n📈 Resultados del filtrado:`);
    console.log(`  🆕 Filas nuevas únicas: ${datosNuevosUnicos.length}`);
    console.log(`  🚫 Duplicados con datos existentes: ${resultadoFiltrado.totalDuplicados}`);
    console.log(`  🔄 Duplicados internos: ${resultadoFiltrado.duplicadosInternos}`);
    
    if (datosNuevosUnicos.length === 0) {
        console.log('\n⚠️ No hay datos nuevos únicos para añadir. El parquet no se modifica.');
        return;
    }
    
    // Combinar datos existentes + nuevos únicos
    const todosCombinados = [...datosExistentes, ...datosNuevosUnicos];
    
    console.log(`\n💾 Guardando ${todosCombinados.length} filas totales en: ${OUTPUT_PATH}`);
    console.log(`  📚 Filas existentes preservadas: ${datosExistentes.length}`);
    console.log(`  🆕 Filas nuevas añadidas: ${datosNuevosUnicos.length}`);
    
    const exitoEscritura = await escribirParquet(todosCombinados, OUTPUT_PATH);
    
    if (exitoEscritura) {
        console.log(`✅ Parquet actualizado exitosamente!`);
        
        console.log(`\n🎉 Proceso completado exitosamente!`);
        console.log(`📊 Estadísticas finales:`);
        console.log(`  📈 Total filas en parquet: ${todosCombinados.length}`);
        console.log(`  📁 Carpetas procesadas: ${totalCarpetasProcesadas}`);
        console.log(`  📄 Archivos xlsx procesados: ${totalArchivosExitosos}`);
        console.log(`  🆕 Filas nuevas añadidas: ${datosNuevosUnicos.length}`);
        console.log(`  🚫 Duplicados evitados: ${resultadoFiltrado.totalDuplicados + resultadoFiltrado.duplicadosInternos}`);
        console.log(`  💾 Archivo parquet: ${OUTPUT_PATH}`);
        
        // Mostrar resumen por partido de los datos nuevos
        if (datosNuevosUnicos.length > 0) {
            console.log(`\n📈 Resumen de datos nuevos por partido:`);
            const resumenNuevos = {};
            datosNuevosUnicos.forEach(row => {
                const key = `${row.Jornada || row.jornada} | ${row.Partido || row.partido} | ${row.tipo_reporte}`;
                resumenNuevos[key] = (resumenNuevos[key] || 0) + 1;
            });
            
            Object.entries(resumenNuevos).forEach(([partidoTipo, count]) => {
                console.log(`  ⚡ ${partidoTipo}: ${count} filas`);
            });
        }
        
    } else {
        console.log(`\n❌ Error al escribir el parquet.`);
    }
}

// Verificar si es el archivo principal
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { 
    procesarArchivoXlsx, 
    extraerJornadaPartido,
    buscarArchivosXlsxEnCarpeta,
    escribirParquet, 
    leerDatosParquetExistentes, 
    generarClaveUnica, 
    filtrarDatosDuplicados 
};