import { Sprout, MapPin, Layers } from 'lucide-react';
import type { OrchardRecord } from '../data/orchards';

interface SidebarProps {
  orchards: OrchardRecord[];
  selectedId: string;
  onSelectOrchard: (id: string) => void;
  selectedOrchard: OrchardRecord;
}

export function Sidebar({
  orchards,
  selectedId,
  onSelectOrchard,
  selectedOrchard,
}: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="Panel de navegación de huertos">
      <div className="sidebar-brand">
        <Sprout size={20} color="#7fd185" />
        <span>Avocado_Predictor</span>
      </div>

      <div className="sidebar-section">
        <label htmlFor="orchard-select" className="sidebar-label">
          Seleccionar huerto (prueba)
        </label>
        <select
          id="orchard-select"
          className="sidebar-select"
          value={selectedId}
          onChange={(e) => onSelectOrchard(e.target.value)}
        >
          {orchards.map((orchard) => (
            <option key={orchard.id} value={orchard.id}>
              {orchard.name}
            </option>
          ))}
        </select>
      </div>

      <div className="sidebar-section">
        <span className="sidebar-label">Ficha del huerto</span>
        <div className="block-item" style={{ backgroundColor: 'var(--sidebar-surface)', borderColor: '#324738', color: 'var(--sidebar-text)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <MapPin size={16} color="#7fd185" />
            <span>{selectedOrchard.municipality}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={16} color="#7fd185" />
            <span>{selectedOrchard.areaHa} ha registradas</span>
          </div>
        </div>
      </div>

      <div className="sidebar-section" style={{ marginTop: 'auto', fontSize: '0.8rem', color: 'var(--sidebar-muted)' }}>
        <span>Modo local · Datos ficticios</span>
        <span>Sin subida de archivos ni LLM</span>
      </div>
    </aside>
  );
}