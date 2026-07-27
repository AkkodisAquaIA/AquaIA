"use client";

import { useState, useRef, useEffect } from "react";
import {
  Microscope, Users, LayoutDashboard, Search, CheckSquare,
  Database, Download, Settings, BookOpen, FolderOpen,
  Upload, FileText, Shield, Tag, Crop, ChevronRight,
  AlertTriangle, Info, Lightbulb, CheckCircle, ArrowRight,
  Package, BarChart3, Lock, Eye, RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ── Section registry ──────────────────────────────────────────────────────────

const SECTIONS = [
  { id: "intro",      label: "Introduction",         icon: Microscope,     color: "text-green-400",   bg: "bg-green-500/10",   border: "border-green-500/20" },
  { id: "workspaces", label: "Espaces de travail",   icon: Users,          color: "text-violet-400",  bg: "bg-violet-500/10",  border: "border-violet-500/20" },
  { id: "dashboard",  label: "Tableau de bord",      icon: LayoutDashboard,color: "text-sky-400",     bg: "bg-sky-500/10",     border: "border-sky-500/20" },
  { id: "search",     label: "Recherche d'images",   icon: Search,         color: "text-amber-400",   bg: "bg-amber-500/10",   border: "border-amber-500/20" },
  { id: "validation", label: "File de validation",   icon: CheckSquare,    color: "text-orange-400",  bg: "bg-orange-500/10",  border: "border-orange-500/20" },
  { id: "dataset",    label: "Dataset Explorer",     icon: Database,       color: "text-indigo-400",  bg: "bg-indigo-500/10",  border: "border-indigo-500/20" },
  { id: "export",     label: "Export Center",        icon: Download,       color: "text-teal-400",    bg: "bg-teal-500/10",    border: "border-teal-500/20" },
  { id: "settings",   label: "Paramètres",           icon: Settings,       color: "text-slate-400",   bg: "bg-slate-500/10",   border: "border-slate-500/20" },
];

// ── Shared sub-components ─────────────────────────────────────────────────────

function SectionHeader({ id, label, icon: Icon, color, bg, border }: typeof SECTIONS[0]) {
  return (
    <div id={id} className={cn("flex items-center gap-3 px-4 py-3 rounded-xl border mb-6", bg, border)}>
      <div className={cn("p-2 rounded-lg", bg, border, "border")}>
        <Icon className={cn("w-5 h-5", color)} />
      </div>
      <h2 className={cn("text-lg font-bold", color)}>{label}</h2>
    </div>
  );
}

function Tip({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 bg-green-500/8 border border-green-500/20 rounded-xl px-4 py-3 my-4">
      <Lightbulb className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />
      <p className="text-sm text-[var(--text-dim)]">{children}</p>
    </div>
  );
}

function Warning({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 bg-amber-500/8 border border-amber-500/20 rounded-xl px-4 py-3 my-4">
      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
      <p className="text-sm text-[var(--text-dim)]">{children}</p>
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 bg-sky-500/8 border border-sky-500/20 rounded-xl px-4 py-3 my-4">
      <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
      <p className="text-sm text-[var(--text-dim)]">{children}</p>
    </div>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-xs bg-[var(--bg-input)] border border-[var(--border)] rounded px-1.5 py-0.5 text-amber-400">
      {children}
    </code>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="font-mono text-xs bg-[var(--bg-input)] border border-[var(--border)] rounded-xl p-4 my-3 overflow-x-auto text-[var(--text-dim)] leading-relaxed">
      {children}
    </pre>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-4 my-3">
      <div className="flex items-center justify-center w-7 h-7 rounded-full bg-[var(--bg-input)] border border-[var(--border)] text-xs font-bold text-[var(--text-dim)] shrink-0 mt-0.5">
        {n}
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-[var(--text-sub)] mb-1">{title}</p>
        <p className="text-sm text-[var(--text-dim)]">{children}</p>
      </div>
    </div>
  );
}

function Badge({ children, color = "default" }: { children: React.ReactNode; color?: "green" | "amber" | "red" | "indigo" | "default" }) {
  const styles = {
    green:   "bg-green-500/10 border-green-500/20 text-green-400",
    amber:   "bg-amber-500/10 border-amber-500/20 text-amber-400",
    red:     "bg-red-500/10  border-red-500/20  text-red-400",
    indigo:  "bg-indigo-500/10 border-indigo-500/20 text-indigo-400",
    default: "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)]",
  };
  return (
    <span className={cn("inline-flex items-center px-2 py-0.5 text-[10px] font-medium rounded-full border", styles[color])}>
      {children}
    </span>
  );
}

// ── Workflow diagram SVG ───────────────────────────────────────────────────────

function WorkflowDiagram() {
  const steps = [
    { label: "Recherche", sub: "iNaturalist & sources", icon: "🔍", color: "#f59e0b" },
    { label: "Validation", sub: "Approuver / Rejeter", icon: "✅", color: "#f97316" },
    { label: "Dataset", sub: "Organiser par espèce", icon: "📁", color: "#6366f1" },
    { label: "Export", sub: "ML-ready ZIP", icon: "📦", color: "#14b8a6" },
  ];
  return (
    <div className="flex items-center gap-2 my-6 overflow-x-auto pb-2">
      {steps.map((s, i) => (
        <div key={s.label} className="flex items-center gap-2 shrink-0">
          <div className="flex flex-col items-center gap-1.5 px-4 py-3 rounded-xl border"
            style={{ background: `${s.color}10`, borderColor: `${s.color}30` }}>
            <span className="text-2xl">{s.icon}</span>
            <span className="text-xs font-semibold" style={{ color: s.color }}>{s.label}</span>
            <span className="text-[10px] text-[var(--text-muted)] text-center max-w-[80px]">{s.sub}</span>
          </div>
          {i < steps.length - 1 && <ArrowRight className="w-4 h-4 text-[var(--text-ghost)] shrink-0" />}
        </div>
      ))}
    </div>
  );
}

// ── Attribution file diagram ───────────────────────────────────────────────────

function AttributionDiagram() {
  const images = ["765.jpg", "766.jpg", "767.jpg"];
  const attrs = [
    { url: "https://www.inaturalist.org/observations/62286468", author: "nmacelko2", license: "CC0 1.0" },
    { url: "https://www.inaturalist.org/observations/244642513", author: "Joseph Aubert", license: "CC BY 4.0" },
    { url: "https://www.inaturalist.org/observations/312847291", author: "Marie Dupont", license: "CC BY-NC 4.0" },
  ];
  return (
    <div className="my-4 flex gap-4 flex-wrap">
      {/* Images list */}
      <div className="flex-1 min-w-[180px]">
        <p className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">Images (ordre alphabétique)</p>
        <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-xl overflow-hidden">
          {images.map((img, i) => (
            <div key={img} className={cn("flex items-center gap-2 px-3 py-1.5", i > 0 && "border-t border-[var(--border)]/50")}>
              <span className="text-[10px] text-[var(--text-muted)] w-4 text-right shrink-0">{i + 1}</span>
              <span className="text-[10px] font-mono text-[var(--text-dim)]">{img}</span>
            </div>
          ))}
        </div>
      </div>
      {/* Arrow */}
      <div className="flex items-center self-center">
        <div className="flex flex-col items-center gap-1">
          <div className="w-8 h-px bg-[var(--border)]" />
          <span className="text-[9px] text-[var(--text-muted)]">associé</span>
          <div className="w-8 h-px bg-[var(--border)]" />
        </div>
      </div>
      {/* Attribution file */}
      <div className="flex-[2] min-w-[260px]">
        <p className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">Fichier .txt (même ordre)</p>
        <div className="bg-[var(--bg-input)] border border-amber-500/20 rounded-xl overflow-hidden">
          {attrs.map((a, i) => (
            <div key={i} className={cn("flex items-start gap-2 px-3 py-1.5", i > 0 && "border-t border-[var(--border)]/50")}>
              <span className="text-[10px] text-[var(--text-muted)] w-4 text-right shrink-0 mt-0.5">{i + 1}</span>
              <div className="min-w-0">
                <p className="text-[10px] font-mono text-amber-400 truncate">{a.url}</p>
                <p className="text-[10px] text-[var(--text-muted)]">{a.author} — <span className="text-green-400">{a.license}</span></p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[10px] text-[var(--text-muted)] mt-1.5 font-mono text-center">url - auteur - licence</p>
      </div>
    </div>
  );
}

// ── Export formats table ───────────────────────────────────────────────────────

function ExportFormatsTable() {
  const formats = [
    { id: "classification", label: "Classification", icon: "🗂️", use: "PyTorch ImageFolder, Keras", structure: "espece/image.jpg", sl: "Par espèce" },
    { id: "yolo",           label: "YOLO",           icon: "🎯", use: "YOLOv5/v8/v11 detection",   structure: "images/espece/",   sl: "Par espèce" },
    { id: "coco",           label: "COCO JSON",      icon: "🔵", use: "Detectron2, MMDetection",   structure: "images/ + .json", sl: "Racine" },
    { id: "csv",            label: "Métadonnées CSV",icon: "📋", use: "Analyse, audit, tableur",   structure: "metadata.csv",    sl: "Racine" },
  ];
  return (
    <div className="overflow-x-auto my-4">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-[var(--border)]">
            {["Format", "Usage typique", "Structure", "Fichier SL"].map(h => (
              <th key={h} className="text-left px-3 py-2 text-[var(--text-muted)] font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {formats.map((f, i) => (
            <tr key={f.id} className={cn("border-b border-[var(--border)]/50", i % 2 === 0 && "bg-[var(--bg-alt)]/50")}>
              <td className="px-3 py-2 font-medium text-[var(--text-sub)]">{f.icon} {f.label}</td>
              <td className="px-3 py-2 text-[var(--text-dim)]">{f.use}</td>
              <td className="px-3 py-2 font-mono text-amber-400/80">{f.structure}</td>
              <td className="px-3 py-2 text-[var(--text-dim)]">{f.sl}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Folder structure illustration ──────────────────────────────────────────────

function FolderStructure() {
  const tree = [
    { indent: 0, name: "Paracorixa_classification.zip", icon: "📦", color: "text-teal-400" },
    { indent: 1, name: "Paracorixa_concinna/", icon: "📁", color: "text-indigo-400" },
    { indent: 2, name: "1234.jpg", icon: "🖼️", color: "text-[var(--text-dim)]" },
    { indent: 2, name: "1235.jpg", icon: "🖼️", color: "text-[var(--text-dim)]" },
    { indent: 2, name: "SL_PARACORIXA_CONCINNA.txt", icon: "📄", color: "text-amber-400" },
    { indent: 1, name: "Sigara_striata/", icon: "📁", color: "text-indigo-400" },
    { indent: 2, name: "5678.jpg", icon: "🖼️", color: "text-[var(--text-dim)]" },
    { indent: 2, name: "SL_SIGARA_STRIATA.txt", icon: "📄", color: "text-amber-400" },
  ];
  return (
    <div className="bg-[var(--bg-input)] border border-[var(--border)] rounded-xl p-4 my-4 font-mono text-xs">
      {tree.map((item, i) => (
        <div key={i} className={cn("flex items-center gap-1.5 py-0.5", item.color)} style={{ paddingLeft: item.indent * 20 }}>
          <span>{item.icon}</span>
          <span>{item.name}</span>
        </div>
      ))}
    </div>
  );
}

// ── Section contents ───────────────────────────────────────────────────────────

function SectionIntro() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        <strong className="text-[var(--text-sub)]">ADIAB</strong> (AquaIA Dataset Builder) est une plateforme professionnelle dédiée à la construction de datasets d'images d'invertébrés aquatiques pour l'entraînement de modèles d'intelligence artificielle.
      </p>

      <div className="grid grid-cols-2 gap-3 my-4">
        {[
          { icon: Search, label: "Collecte",  desc: "Recherche automatisée sur iNaturalist et autres sources", color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
          { icon: CheckSquare, label: "Validation", desc: "Contrôle qualité manuel image par image", color: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/20" },
          { icon: Database, label: "Organisation", desc: "Datasets structurés par espèce avec attribution", color: "text-indigo-400", bg: "bg-indigo-500/10", border: "border-indigo-500/20" },
          { icon: Package, label: "Export", desc: "ZIP prêt-à-l'emploi pour PyTorch, YOLO, COCO", color: "text-teal-400", bg: "bg-teal-500/10", border: "border-teal-500/20" },
        ].map(({ icon: Icon, label, desc, color, bg, border }) => (
          <div key={label} className={cn("rounded-xl border p-3", bg, border)}>
            <div className="flex items-center gap-2 mb-1">
              <Icon className={cn("w-4 h-4", color)} />
              <span className={cn("text-sm font-semibold", color)}>{label}</span>
            </div>
            <p className="text-xs text-[var(--text-dim)]">{desc}</p>
          </div>
        ))}
      </div>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-6 mb-2">Flux de travail général</h3>
      <WorkflowDiagram />

      <Note>
        ADIAB est conçu pour les projets scientifiques nécessitant une traçabilité complète des sources d'images. Chaque image conserve ses métadonnées d'attribution (URL source, auteur, licence) tout au long du processus.
      </Note>
    </div>
  );
}

function SectionWorkspaces() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        Les <strong className="text-[var(--text-sub)]">espaces de travail</strong> permettent à plusieurs utilisateurs ou projets de coexister sur la même instance ADIAB, chacun avec ses propres images, datasets et exports.
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-4 mb-3">Créer un espace de travail</h3>
      <Step n={1} title="Ouvrir la liste des espaces">
        Sur l'écran de sélection, cliquer sur <strong>+ Nouvel espace</strong>.
      </Step>
      <Step n={2} title="Choisir un nom">
        Saisir un nom descriptif (ex. : <em>Projet Macroinvertébrés 2025</em>).
      </Step>
      <Step n={3} title="Définir un code PIN (optionnel)">
        Cocher <strong>Protéger par code PIN</strong> pour restreindre l'accès en écriture. Les espaces non protégés sont accessibles en lecture par tous.
      </Step>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Niveaux d'accès</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
          <div className="flex items-center gap-2 mb-1">
            <Eye className="w-4 h-4 text-sky-400" />
            <span className="text-xs font-semibold text-sky-400">Mode lecture</span>
          </div>
          <p className="text-xs text-[var(--text-dim)]">Consultation des images et datasets uniquement. Aucune modification possible.</p>
        </div>
        <div className="rounded-xl border border-green-500/20 p-3 bg-green-500/5">
          <div className="flex items-center gap-2 mb-1">
            <Lock className="w-4 h-4 text-green-400" />
            <span className="text-xs font-semibold text-green-400">Mode complet</span>
          </div>
          <p className="text-xs text-[var(--text-dim)]">Accès en écriture complet après authentification par code PIN.</p>
        </div>
      </div>

      <Tip>
        Pour les espaces partagés en équipe, activez la protection par PIN. Les collaborateurs peuvent consulter les datasets sans PIN mais ne peuvent pas modifier.
      </Tip>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-2">Changer d'espace</h3>
      <p className="text-sm text-[var(--text-dim)]">
        Cliquer sur le nom de l'espace en haut à gauche de l'interface pour revenir à l'écran de sélection.
      </p>
    </div>
  );
}

function SectionDashboard() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        Le <strong className="text-[var(--text-sub)]">tableau de bord</strong> affiche une vue d'ensemble de l'état de votre espace de travail en temps réel.
      </p>

      <div className="grid grid-cols-2 gap-3 my-4">
        {[
          { icon: "🖼️", label: "Images totales",       desc: "Nombre total d'images dans l'espace" },
          { icon: "✅", label: "Images validées",       desc: "Prêtes pour l'export en dataset" },
          { icon: "⏳", label: "En attente",            desc: "Dans la file de validation" },
          { icon: "📁", label: "Datasets créés",        desc: "Collections organisées par projet" },
        ].map(({ icon, label, desc }) => (
          <div key={label} className="rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{icon}</span>
              <span className="text-xs font-semibold text-[var(--text-sub)]">{label}</span>
            </div>
            <p className="text-xs text-[var(--text-dim)]">{desc}</p>
          </div>
        ))}
      </div>

      <Note>
        Les statistiques se mettent à jour automatiquement. Aucun rechargement manuel n'est nécessaire.
      </Note>
    </div>
  );
}

function SectionSearch() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        Le module de <strong className="text-[var(--text-sub)]">recherche</strong> permet de collecter automatiquement des images depuis des bases de données scientifiques en ligne.
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-4 mb-3">Lancer une recherche</h3>
      <Step n={1} title="Saisir le nom d'espèce">
        Entrer le nom scientifique de l'espèce (ex. : <em>Baetis rhodani</em>). La complétion automatique propose des suggestions taxonomiques.
      </Step>
      <Step n={2} title="Sélectionner les sources">
        Choisir les bases de données à interroger : <Badge>iNaturalist</Badge> et autres sources disponibles.
      </Step>
      <Step n={3} title="Définir le nombre d'images">
        Ajuster la limite (ex. : 50, 100) selon vos besoins. Les résultats sont ordonnés par pertinence.
      </Step>
      <Step n={4} title="Lancer et attendre">
        Les images trouvées sont automatiquement ajoutées à la file de validation avec leurs métadonnées d'attribution complètes.
      </Step>

      <Tip>
        Les images récupérées conservent automatiquement l'URL source, le nom de l'auteur et la licence — ces informations sont essentielles pour la traçabilité légale du dataset.
      </Tip>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-2">Historique</h3>
      <p className="text-sm text-[var(--text-dim)]">
        Toutes les recherches effectuées sont conservées dans l'historique avec la date, la requête et le nombre de résultats obtenus.
      </p>
    </div>
  );
}

function SectionValidation() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        La <strong className="text-[var(--text-sub)]">file de validation</strong> est l'étape de contrôle qualité où chaque image est examinée avant d'intégrer le dataset.
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-4 mb-3">Actions disponibles</h3>
      <div className="space-y-2">
        {[
          { icon: CheckCircle, color: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20", action: "Valider", desc: "L'image est acceptée et passe dans le Dataset Explorer avec le statut « validé »." },
          { icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20", action: "Rejeter", desc: "L'image est écartée et ne sera pas incluse dans les exports." },
          { icon: Crop, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", action: "Recadrer", desc: "Rogner l'image pour isoler le spécimen avant validation. Le recadrage est non-destructif." },
          { icon: Tag, color: "text-indigo-400", bg: "bg-indigo-500/10", border: "border-indigo-500/20", action: "Attribuer une espèce", desc: "Assigner l'image à un taxon. Obligatoire avant que l'image puisse rejoindre un dossier espèce." },
        ].map(({ icon: Icon, color, bg, border, action, desc }) => (
          <div key={action} className={cn("flex items-start gap-3 rounded-xl border p-3", bg, border)}>
            <Icon className={cn("w-4 h-4 mt-0.5 shrink-0", color)} />
            <div>
              <p className={cn("text-xs font-semibold", color)}>{action}</p>
              <p className="text-xs text-[var(--text-dim)] mt-0.5">{desc}</p>
            </div>
          </div>
        ))}
      </div>

      <Warning>
        Une image sans espèce assignée ne peut pas être ajoutée au Dataset Explorer. Toujours attribuer un taxon avant de valider.
      </Warning>
    </div>
  );
}

function SectionDataset() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        Le <strong className="text-[var(--text-sub)]">Dataset Explorer</strong> organise toutes les images validées en <em>dossiers d'espèces</em> et en <em>datasets</em> (collections thématiques multi-espèces).
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-4 mb-2">Structure en dossiers d'espèces</h3>
      <p className="text-sm text-[var(--text-dim)]">
        Chaque espèce possède son propre dossier nommé d'après le nom scientifique (ex. : <Code>Paracorixa concinna</Code>). Une image ne peut pas exister en dehors d'un dossier espèce.
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Uploader des images locales</h3>
      <p className="text-sm text-[var(--text-dim)] mb-3">
        Pour ajouter des images locales directement dans le Dataset Explorer (sans passer par la file de validation), utiliser le bouton <strong>Upload images</strong>. Le processus se fait en 4 étapes :
      </p>

      <Step n={1} title="Glisser le dossier">
        Faire glisser le dossier contenant les images depuis le Finder / Explorateur directement sur la zone de dépôt. ADIAB détecte automatiquement les images et le fichier d'attribution <Code>.txt</Code>.
      </Step>
      <Step n={2} title="Vérifier le fichier d'attribution">
        Le fichier <Code>.txt</Code> trouvé dans le dossier est chargé automatiquement. Vérifier que le nombre de lignes correspond au nombre d'images.
      </Step>
      <Step n={3} title="Sélectionner le dossier espèce">
        Choisir un dossier existant ou créer une nouvelle espèce.
      </Step>
      <Step n={4} title="Assigner à un dataset (optionnel)">
        Optionnellement assigner les images à un dataset existant au moment de l'upload.
      </Step>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Format du fichier d'attribution</h3>
      <p className="text-sm text-[var(--text-dim)] mb-2">
        Chaque image doit être associée à une ligne dans le fichier <Code>.txt</Code>. Les images sont triées alphabétiquement ; le fichier doit suivre le même ordre.
      </p>

      <CodeBlock>{`https://www.inaturalist.org/observations/62286468 - nmacelko2 - CC0 1.0
https://www.inaturalist.org/observations/244642513 - Joseph Aubert - CC BY 4.0
https://www.inaturalist.org/observations/312847291 - Marie Dupont - CC BY-NC 4.0`}</CodeBlock>

      <p className="text-xs text-[var(--text-muted)] -mt-1 mb-3">Format : <Code>url_source - auteur - licence</Code> · séparateur : <Code> - </Code> (espace tiret espace)</p>

      <AttributionDiagram />

      <Warning>
        Si le nombre de lignes du fichier <Code>.txt</Code> ne correspond pas au nombre d'images, le bouton Upload reste désactivé. Corriger le fichier avant de continuer.
      </Warning>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Gérer les datasets</h3>
      <div className="space-y-2">
        {[
          { icon: FolderOpen, color: "text-indigo-400", action: "Créer un dataset",   desc: "Cliquer sur + Nouveau dataset. Un dataset regroupe plusieurs espèces pour un projet ML spécifique." },
          { icon: ChevronRight, color: "text-green-400", action: "Assigner une espèce", desc: "Dans le header du dossier espèce, cliquer sur le menu « Ajouter au dataset » et sélectionner le dataset cible." },
          { icon: RefreshCw, color: "text-amber-400",  action: "Renommer un dataset", desc: "Cliquer sur l'icône crayon à côté du nom du dataset pour le renommer directement." },
        ].map(({ icon: Icon, color, action, desc }) => (
          <div key={action} className="flex items-start gap-3 rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
            <Icon className={cn("w-4 h-4 mt-0.5 shrink-0", color)} />
            <div>
              <p className="text-xs font-semibold text-[var(--text-sub)]">{action}</p>
              <p className="text-xs text-[var(--text-dim)] mt-0.5">{desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionExport() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        L'<strong className="text-[var(--text-sub)]">Export Center</strong> génère des archives ZIP prêtes à l'emploi pour l'entraînement de modèles ML, avec les métadonnées d'attribution incluses.
      </p>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-4 mb-3">Créer un export</h3>
      <Step n={1} title="Sélectionner la source">
        Choisir entre <Badge color="indigo">Toutes les images validées</Badge> ou un <Badge color="indigo">Dataset spécifique</Badge>.
      </Step>
      <Step n={2} title="Choisir le format">
        Sélectionner le format adapté à votre framework ML (voir tableau ci-dessous).
      </Step>
      <Step n={3} title="Lancer l'export">
        L'export se génère en arrière-plan. L'état passe de <Badge color="amber">En cours</Badge> à <Badge color="green">Terminé</Badge> automatiquement.
      </Step>
      <Step n={4} title="Télécharger">
        Cliquer sur le bouton de téléchargement. Le fichier est nommé <Code>NomDataset_format.zip</Code>.
      </Step>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Formats disponibles</h3>
      <ExportFormatsTable />

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-2">Structure d'un export Classification</h3>
      <FolderStructure />

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-3">Fichiers SL (Sources & Licences)</h3>
      <p className="text-sm text-[var(--text-dim)] mb-3">
        Chaque export inclut des fichiers <Code>SL_NOM_ESPECE.txt</Code> qui documentent l'attribution légale de chaque image. Ces fichiers sont obligatoires pour la conformité aux licences Creative Commons.
      </p>

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
          <p className="text-xs font-semibold text-amber-400 mb-1">📄 Nommage</p>
          <p className="text-xs font-mono text-[var(--text-dim)]">SL_NOM_ESPECE.txt</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Nom en MAJUSCULES, espaces remplacés par _</p>
        </div>
        <div className="rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
          <p className="text-xs font-semibold text-amber-400 mb-1">📋 Format</p>
          <p className="text-xs font-mono text-[var(--text-dim)]">url - auteur - licence</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Une ligne par image, dans l'ordre du dossier</p>
        </div>
      </div>

      <CodeBlock>{`# SL_PARACORIXA_CONCINNA.txt
https://www.inaturalist.org/observations/62286468 - nmacelko2 - CC0 1.0
https://www.inaturalist.org/observations/244642513 - Joseph Aubert - CC BY 4.0`}</CodeBlock>

      <Tip>
        Le nom du fichier ZIP téléchargé reflète automatiquement le nom du dataset sélectionné, par exemple <Code>Paracorixa_classification.zip</Code>.
      </Tip>
    </div>
  );
}

function SectionSettings() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-[var(--text-dim)] leading-relaxed">
        Le panneau <strong className="text-[var(--text-sub)]">Paramètres</strong> regroupe les options de gestion de l'espace de travail.
      </p>

      <div className="space-y-2 mt-4">
        {[
          { icon: Lock, color: "text-violet-400", label: "Code PIN",         desc: "Définir, modifier ou supprimer le code PIN de protection de l'espace." },
          { icon: RefreshCw, color: "text-sky-400", label: "Renommer l'espace", desc: "Modifier le nom d'affichage de l'espace de travail." },
          { icon: BarChart3, color: "text-green-400", label: "Statistiques",    desc: "Vue détaillée des compteurs : images par statut, espèces, datasets, exports." },
          { icon: Shield, color: "text-red-400",    label: "Supprimer",       desc: "Suppression définitive de l'espace et de toutes ses données. Action irréversible." },
        ].map(({ icon: Icon, color, label, desc }) => (
          <div key={label} className="flex items-start gap-3 rounded-xl border border-[var(--border)] p-3 bg-[var(--bg-alt)]">
            <Icon className={cn("w-4 h-4 mt-0.5 shrink-0", color)} />
            <div>
              <p className="text-xs font-semibold text-[var(--text-sub)]">{label}</p>
              <p className="text-xs text-[var(--text-dim)] mt-0.5">{desc}</p>
            </div>
          </div>
        ))}
      </div>

      <h3 className="text-sm font-semibold text-[var(--text-sub)] mt-5 mb-2">Thème</h3>
      <p className="text-sm text-[var(--text-dim)]">
        Basculer entre le mode clair et le mode sombre via l'icône soleil/lune en bas du menu de navigation.
      </p>
    </div>
  );
}

const SECTION_CONTENT: Record<string, React.ReactNode> = {
  intro:      <SectionIntro />,
  workspaces: <SectionWorkspaces />,
  dashboard:  <SectionDashboard />,
  search:     <SectionSearch />,
  validation: <SectionValidation />,
  dataset:    <SectionDataset />,
  export:     <SectionExport />,
  settings:   <SectionSettings />,
};

// ── Main panel ─────────────────────────────────────────────────────────────────

export default function DocsPanel() {
  const [active, setActive] = useState("intro");
  const contentRef = useRef<HTMLDivElement>(null);

  const activeMeta = SECTIONS.find(s => s.id === active)!;

  useEffect(() => {
    contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [active]);

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── TOC sidebar ── */}
      <aside className="w-52 shrink-0 border-r flex flex-col overflow-y-auto"
        style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
        <div className="px-4 py-4 border-b" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-green-400" />
            <span className="text-sm font-bold text-[var(--text-base)]">Documentation</span>
          </div>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Guide complet ADIAB</p>
        </div>
        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {SECTIONS.map(({ id, label, icon: Icon, color, bg, border }) => (
            <button key={id} onClick={() => setActive(id)}
              className={cn(
                "flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-xs transition-all text-left",
                active === id
                  ? cn("border font-semibold", bg, border, color)
                  : "border border-transparent text-[var(--text-dim)] hover:bg-[var(--bg-input)]"
              )}>
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span className="leading-tight">{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* ── Content area ── */}
      <main ref={contentRef} className="flex-1 overflow-y-auto p-6"
        style={{ background: "var(--bg-base)" }}>
        <div className="max-w-2xl mx-auto">
          <SectionHeader {...activeMeta} />
          {SECTION_CONTENT[active]}
          {/* Navigation buttons */}
          <div className="flex items-center justify-between mt-8 pt-5 border-t" style={{ borderColor: "var(--border)" }}>
            {(() => {
              const idx = SECTIONS.findIndex(s => s.id === active);
              const prev = SECTIONS[idx - 1];
              const next = SECTIONS[idx + 1];
              return (
                <>
                  {prev ? (
                    <button onClick={() => setActive(prev.id)}
                      className="flex items-center gap-2 text-xs text-[var(--text-dim)] hover:text-[var(--text-sub)] transition-colors">
                      <ChevronRight className="w-3.5 h-3.5 rotate-180" />
                      {prev.label}
                    </button>
                  ) : <div />}
                  {next && (
                    <button onClick={() => setActive(next.id)}
                      className="flex items-center gap-2 text-xs text-[var(--text-dim)] hover:text-[var(--text-sub)] transition-colors">
                      {next.label}
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  )}
                </>
              );
            })()}
          </div>
        </div>
      </main>
    </div>
  );
}
