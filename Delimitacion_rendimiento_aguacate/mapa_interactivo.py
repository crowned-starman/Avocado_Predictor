"""Mapa HTML autónomo para revisar municipios y exportar su selección."""

import html
import json


def crear_mapa(municipios, resumen, seleccion, salida, meses):
    detalles = {
        str(f.CVEGEO): dict(
            nombre=f.municipio_inegi,
            anios=int(f.anios_con_etiqueta),
            superficie=float(f.superficie_cosechada_ha_anio_promedio),
        )
        for f in resumen.itertuples()
    }
    paths = []
    for c, n, g in municipios:
        d = []
        geoms = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for pol in geoms:
            pol = pol.simplify(0.0007, preserve_topology=True)
            for anillo in [pol.exterior, *pol.interiors]:
                coords = list(anillo.coords)
                d.append("M" + " L".join(f"{x:.6f},{-y:.6f}" for x, y in coords) + " Z")
        clase = (
            "candidato elegido"
            if c in seleccion
            else ("candidato" if c in detalles else "sin-datos")
        )
        paths.append(
            f'<path data-cve="{c}" class="{clase}" d="{" ".join(d)}"><title>{html.escape(n)} ({c})</title></path>'
        )
    texto = """<!doctype html>
<html lang="es"><meta charset="utf-8"><title>Seleccionar municipios de aguacate</title>
<style>
body{font:16px system-ui;margin:20px;color:#18372a;background:#f7faf7}h1{font-size:24px;margin-bottom:5px}
main{display:grid;grid-template-columns:minmax(400px,2fr) minmax(260px,1fr);gap:20px}
svg{width:100%;height:620px;background:white;border:1px solid #cbd8cc;touch-action:none}
path{stroke:#68786c;stroke-width:.002;fill:#e8ece9;fill-rule:evenodd;vector-effect:non-scaling-stroke;stroke-width:.7px}
.candidato{fill:#d8e8bd;cursor:pointer}.elegido{fill:#528747}.candidato:hover{fill:#e2b65c}
button,input{font:inherit;padding:9px;margin:4px;border:1px solid #a4bba6;border-radius:5px;background:white}
button{cursor:pointer}button.principal{background:#235f43;color:white}#lista{height:330px;overflow:auto;background:white;padding:8px}
#lista label{display:block;font-size:14px}small{line-height:1.5}#detalle{min-height:65px;font-weight:600}
@media(max-width:800px){main{display:block}svg{height:460px}}
</style>
<h1>Municipios para la prueba de rendimiento de aguacate</h1>
<p>Haz clic en los municipios de tu franja central. Verde oscuro = elegido; verde claro = candidato con cosecha 2020–2025; gris = sin etiqueta disponible.</p>
<main><section><svg id="mapa" viewBox="-103.9 -20.65 4.1 2.9">__PATHS__</svg>
<button id="mas">Acercar +</button><button id="menos">Alejar −</button><button id="reset">Restablecer vista</button>
<p><small>Arrastra para mover el mapa. Geometría simplificada solo para esta vista. Los archivos GeoJSON conservan el detalle geográfico.</small></p></section>
<aside><div id="detalle">Pasa el puntero sobre un municipio para ver su nombre.</div>
<p id="conteo"></p><input id="buscar" placeholder="Buscar municipio" style="width:85%">
<div><button id="todos">Todos los candidatos</button><button id="vaciar">Vaciar</button></div>
<div id="lista"></div><button id="descargar" class="principal">Guardar selección para Python</button>
<p><small>El mapa descarga <b>seleccion_municipios.json</b>. Aplícalo desde la raíz con <b>python ejecutar.py aplicar-seleccion --archivo RUTA_DEL_JSON</b> y después ejecuta <b>python ejecutar.py territorio</b>. La selección debe contener al menos un municipio.</small></p>
<p><small>La zona municipal incluye bosques, ciudades y otros cultivos. Este mapa no identifica parcelas de aguacate. Datos: INEGI, Marco Geoestadístico diciembre 2025; DGSIAP, cierres 2020–2025. Selección basada exclusivamente en años de desarrollo.</small></p></aside></main>
<script>
const datos=__DATOS__, elegidos=new Set(__SELECCION__), meses=__MESES__;
const svg=document.querySelector('#mapa'), lista=document.querySelector('#lista');
function refrescar(){
 document.querySelectorAll('path.candidato').forEach(p=>p.classList.toggle('elegido',elegidos.has(p.dataset.cve)));
 document.querySelector('#conteo').textContent=elegidos.size+' municipios elegidos de '+Object.keys(datos).length+' candidatos';
 lista.innerHTML='';const filtro=document.querySelector('#buscar').value.toLocaleLowerCase('es');
 Object.entries(datos).sort((a,b)=>a[1].nombre.localeCompare(b[1].nombre,'es')).forEach(([c,d])=>{
  if(!d.nombre.toLocaleLowerCase('es').includes(filtro))return;
  const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.checked=elegidos.has(c);
  i.onchange=()=>{if(i.checked)elegidos.add(c);else elegidos.delete(c);refrescar()};l.append(i,document.createTextNode(d.nombre+' · '+c));lista.append(l);
 });
}
svg.querySelectorAll('path').forEach(p=>{
 p.onmouseenter=()=>{const d=datos[p.dataset.cve];document.querySelector('#detalle').textContent=d?
 d.nombre+' · '+d.anios+' años con etiqueta · '+Math.round(d.superficie).toLocaleString('es-MX')+' ha cosechadas promedio en años registrados':p.querySelector('title').textContent+' · sin etiqueta en esta selección'};
});
let inicio=null,movido=false;
svg.addEventListener('pointerdown',e=>{inicio={x:e.clientX,y:e.clientY,v:[...svg.getAttribute('viewBox').split(' ').map(Number)],p:e.target};movido=false;svg.setPointerCapture(e.pointerId)});
svg.addEventListener('pointermove',e=>{if(!inicio)return;const r=svg.getBoundingClientRect(),dx=e.clientX-inicio.x,dy=e.clientY-inicio.y;
 if(Math.abs(dx)+Math.abs(dy)>5)movido=true;if(movido){const [x,y,w,h]=inicio.v;svg.setAttribute('viewBox',`${x-dx*w/r.width} ${y-dy*h/r.height} ${w} ${h}`)}});
svg.addEventListener('pointerup',()=>{if(inicio&&!movido){const c=inicio.p.dataset.cve;if(datos[c]){if(elegidos.has(c))elegidos.delete(c);else elegidos.add(c);refrescar()}}inicio=null});
svg.addEventListener('pointercancel',()=>{inicio=null});
function zoom(f){let [x,y,w,h]=svg.getAttribute('viewBox').split(' ').map(Number);svg.setAttribute('viewBox',`${x+w*(1-f)/2} ${y+h*(1-f)/2} ${w*f} ${h*f}`)}
document.querySelector('#mas').onclick=()=>zoom(.75);document.querySelector('#menos').onclick=()=>zoom(1.3);
document.querySelector('#reset').onclick=()=>svg.setAttribute('viewBox','-103.9 -20.65 4.1 2.9');
document.querySelector('#todos').onclick=()=>{Object.keys(datos).forEach(c=>elegidos.add(c));refrescar()};
document.querySelector('#vaciar').onclick=()=>{elegidos.clear();refrescar()};document.querySelector('#buscar').oninput=refrescar;
document.querySelector('#descargar').onclick=()=>{if(!elegidos.size){alert('Elige al menos un municipio.');return}
 const b=new Blob([JSON.stringify({cvegeo_seleccionados:[...elegidos].sort(),meses_emision_propuestos:meses},null,2)],{type:'application/json'});
 const a=document.createElement('a'),u=URL.createObjectURL(b);a.href=u;a.download='seleccion_municipios.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
refrescar();
</script></html>"""
    texto = texto.replace("__PATHS__", "\n".join(paths)).replace(
        "__DATOS__", json.dumps(detalles, ensure_ascii=False).replace("</", "<\\/")
    )
    texto = texto.replace("__SELECCION__", json.dumps(sorted(seleccion))).replace(
        "__MESES__", json.dumps(meses)
    )
    salida.write_text(texto, encoding="utf-8")
