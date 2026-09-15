import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { TooltipProvider } from '@/components/ui/tooltip'
import { 
  Loader2, Sparkles, Zap, Target, Brain, Gauge, Code, GitBranch, AlertTriangle, BookOpen, Server, Layers, 
  Globe, FileText, CheckCircle, ArrowUpRight, Link, X, RefreshCw
} from 'lucide-react'
import './App.css'

const LABELS = [
  'Politics & Government',
  'War & Conflicts',
  'Crime & Justice',
  'Law & Legal',
  'Economy & Business',
  'Science & Technology',
  'Health',
  'Education',
  'Disasters & Emergencies',
  'Sports',
  'Environment & Climate',
]

const LABEL_COLORS = {
  'Politics & Government': '#ef4444',
  'War & Conflicts': '#dc2626',
  'Crime & Justice': '#a855f7',
  'Law & Legal': '#c084fc',
  'Economy & Business': '#22c55e',
  'Science & Technology': '#3b82f6',
  'Health': '#14b8a6',
  'Education': '#f59e0b',
  'Disasters & Emergencies': '#f97316',
  'Sports': '#10b981',
  'Environment & Climate': '#06b6d4',
}

const LABEL_DESCRIPTIONS = {
  'Politics & Government': 'Elections, legislation, policy, political figures',
  'War & Conflicts': 'Armed conflict, military operations, terrorism',
  'Crime & Justice': 'Criminal acts, law enforcement, courts',
  'Law & Legal': 'Civil law, lawsuits, court rulings, regulation',
  'Economy & Business': 'Markets, finance, companies, trade',
  'Science & Technology': 'Research, space, tech products, AI, computing',
  'Health': 'Medical conditions, treatments, pandemics',
  'Education': 'Schools, universities, curricula, admissions',
  'Disasters & Emergencies': 'Natural disasters, accidents, rescue ops',
  'Sports': 'Competitive events, teams, athletes',
  'Environment & Climate': 'Climate change, pollution, conservation',
}

const LABEL_EXAMPLES = {
  'Politics & Government': 'Senate passes infrastructure bill with bipartisan support',
  'War & Conflicts': 'Ukraine forces advance in eastern front after months of stalemate',
  'Crime & Justice': 'Man arrested for armed robbery at downtown bank',
  'Law & Legal': 'Supreme Court hears arguments in major tech antitrust case',
  'Economy & Business': 'Stock market rallies on strong earnings from major tech companies',
  'Science & Technology': 'Scientists discover high-efficiency solar cell material',
  'Health': 'WHO warns of new disease outbreak spreading across Southeast Asia',
  'Education': 'University announces free tuition for all low-income students',
  'Disasters & Emergencies': 'Hurricane makes landfall causing widespread coastal damage',
  'Sports': 'Team wins championship in overtime thriller, securing third title',
  'Environment & Climate': 'Arctic ice melts at record rate, scientists sound alarm',
}

const PER_CLASS_F1 = {
  'Sports': 0.958,
  'Health': 0.870,
  'Politics & Government': 0.866,
  'Crime & Justice': 0.837,
  'Economy & Business': 0.818,
  'Disasters & Emergencies': 0.786,
  'War & Conflicts': 0.772,
  'Science & Technology': 0.723,
  'Environment & Climate': 0.663,
  'Education': 0.601,
  'Law & Legal': 0.537,
}

const STRENGTH_LABELS = {
  'Sports': 'Excellent',
  'Health': 'Very Good',
  'Politics & Government': 'Very Good',
  'Crime & Justice': 'Good',
  'Economy & Business': 'Good',
  'Disasters & Emergencies': 'Good',
  'War & Conflicts': 'Good',
  'Science & Technology': 'Moderate',
  'Environment & Climate': 'Moderate',
  'Education': 'Weak',
  'Law & Legal': 'Weak',
}

const STRENGTH_CLASSES = {
  'Excellent': 'strength-excellent',
  'Very Good': 'strength-very-good',
  'Good': 'strength-good',
  'Moderate': 'strength-moderate',
  'Weak': 'strength-weak',
}

const EXAMPLES = [
  'Senate passes new infrastructure bill with bipartisan support after months of debate',
  'Scientists discover high-efficiency solar cell material that could revolutionize renewable energy',
  'Team wins championship in overtime thriller, securing their third title in five years',
  'Hurricane makes landfall causing widespread damage across coastal communities',
  'Stock market rallies on strong earnings reports from major tech companies',
  'University announces free tuition for all low-income students starting next semester',
  'WHO warns of new disease outbreak spreading rapidly across Southeast Asia',
  'Man arrested for armed robbery at downtown bank, police say suspect had prior record',
  'Supreme Court hears arguments in major tech antitrust case that could reshape industry',
]

const METRICS = [
  { value: '82.4%', label: 'Accuracy' },
  { value: '0.766', label: 'Macro F1' },
  { value: '0.64 GFLOP', label: 'Compute' },
  { value: '11.2M', label: 'Parameters' },
  { value: '44 MB', label: 'Model Size' },
  { value: '~6 ms', label: 'CPU Latency' },
]

function ConfidenceRing({ confidence, color }) {
  const radius = 80
  const stroke = 10
  const normalizedRadius = radius - stroke
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (confidence * circumference)

  return (
    <div className="confidence-ring-container">
      <svg height={radius * 2} width={radius * 2} className="confidence-ring-svg">
        <circle
          stroke="rgba(0,0,0,0.05)"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference + ' ' + circumference}
          style={{
            strokeDashoffset,
            transition: 'stroke-dashoffset 1s ease-in-out',
          }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          transform={`rotate(-90 ${radius} ${radius})`}
        />
      </svg>
      <div className="confidence-ring-text">
        <span className="confidence-ring-number">{(confidence * 100).toFixed(1)}</span>
        <span className="confidence-ring-percent">%</span>
      </div>
    </div>
  )
}

function HeroSection() {
  return (
    <header className="app-header">
      <div className="header-badge">
        <Zap className="size-4" />
        Knowledge Distillation Project
      </div>
      <h1 className="app-title">
        Article Domain Tagger
      </h1>
      <p className="app-subtitle">
        Classify news articles into <strong>11 topic domains</strong> using a 
        <strong>44 MB distilled model</strong> that runs in <strong>~6 ms on CPU</strong> — 
        0.64 GFLOP, 11.2M parameters, 82.4% accuracy
      </p>
    </header>
  )
}

function ClassifierSection({ onClassify, onFetchUrl, text, setText, url, setUrl, loading, error, result, inputMode, setInputMode }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      if (inputMode === 'text') onClassify()
      else onFetchUrl()
    }
  }

  const handleExampleClick = (ex) => {
    setText(ex)
    setInputMode('text')
  }

  const clearUrl = () => {
    setUrl('')
  }

  return (
    <div className="input-section">
      <Card className="card">
        <CardContent className="space-y-4">
          <div className="input-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={inputMode === 'text'}
              className={`input-tab ${inputMode === 'text' ? 'active' : ''}`}
              onClick={() => setInputMode('text')}
            >
              <FileText className="tab-icon" />
              Paste Text
            </button>
            <button
              role="tab"
              aria-selected={inputMode === 'url'}
              className={`input-tab ${inputMode === 'url' ? 'active' : ''}`}
              onClick={() => setInputMode('url')}
            >
              <Globe className="tab-icon" />
              Fetch from URL
            </button>
          </div>

          {inputMode === 'text' && (
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Paste your article text here... (Ctrl+Enter to classify)"
              rows={6}
              className="textarea resize-none text-base"
            />
          )}

          {inputMode === 'url' && (
            <div className="url-input-group">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="https://example.com/news/article"
                className="url-input"
                aria-label="Article URL"
              />
              <Button
                onClick={onFetchUrl}
                disabled={loading || !url.trim()}
                className="btn btn-primary fetch-btn"
              >
                {loading ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Fetching...
                  </>
                ) : (
                  <>
                    <ArrowUpRight className="size-4" />
                    Fetch & Classify
                  </>
                )}
              </Button>
            </div>
          )}

          <div className="input-hint">
            <span>
              {inputMode === 'text' 
                ? 'Tip: Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to classify' 
                : 'Tip: Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to fetch and classify'}
            </span>
            {url && inputMode === 'url' && (
              <button onClick={clearUrl} className="text-primary hover:underline text-sm flex items-center gap-1">
                <X className="size-3" /> Clear
              </button>
            )}
          </div>

          <Button
            onClick={inputMode === 'text' ? onClassify : onFetchUrl}
            disabled={loading || (inputMode === 'text' ? !text.trim() : !url.trim())}
            size="lg"
            className="btn btn-primary btn-lg w-full"
          >
            {loading ? (
              <>
                <Loader2 className="size-5 animate-spin" />
                {inputMode === 'text' ? 'Classifying...' : 'Fetching & Classifying...'}
              </>
            ) : inputMode === 'text' ? (
              <>
                <Sparkles className="size-5" />
                Classify Article
              </>
            ) : (
              <>
                <Globe className="size-5" />
                Fetch & Classify from URL
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div className="examples-section">
        <span className="examples-label">Try an example</span>
        <div className="examples-grid">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              className="example-pill"
              onClick={() => handleExampleClick(ex)}
            >
              <Target className="size-3.5 opacity-60" />
              {ex.slice(0, 70)}...
            </button>
          ))}
        </div>
      </div>

      {error && (
        <Card className="card error-card">
          <CardContent className="py-4 px-6 flex items-center gap-3">
            <AlertTriangle className="size-5 flex-shrink-0" />
            <p className="error-text">{error}</p>
          </CardContent>
        </Card>
      )}

      {result && (
        <div className="results-section fade-in-up">
          {result.extracted_text && (
            <div className="extracted-preview">
              <div className="extracted-preview-header">
                <Globe className="size-5 text-success" />
                <h3 className="extracted-preview-title">Fetched: {result.title}</h3>
              </div>
              <p className="extracted-preview-text">{result.extracted_text.slice(0, 500)}...</p>
            </div>
          )}

          <Card className="card prediction-card">
            <CardContent className="flex flex-col items-center gap-4 py-6">
              <span className="prediction-label">Predicted Domain</span>
              <ConfidenceRing
                confidence={result.confidence}
                color={LABEL_COLORS[result.prediction] || '#2563eb'}
              />
              <h2 className="prediction-domain" style={{ color: LABEL_COLORS[result.prediction] || '#1e293b' }}>
                {result.prediction}
              </h2>
              <p className="prediction-confidence">
                {(result.confidence * 100).toFixed(1)}% model confidence
              </p>
            </CardContent>
          </Card>

          <Card className="card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <CheckCircle className="size-4 text-success" />
                Top 3 Predictions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-0">
              {result.top3.map((item, i) => (
                <div key={i} className="top3-row">
                  <div className="top3-rank" style={{ backgroundColor: LABEL_COLORS[item.label] + '15', color: LABEL_COLORS[item.label] }}>
                    <span className="rank-number">{i + 1}</span>
                  </div>
                  <div className="top3-info">
                    <div className="top3-label-row">
                      <span className="top3-label">{item.label}</span>
                      <span className="top3-confidence" style={{ color: LABEL_COLORS[item.label] }}>
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="top3-bar-bg">
                      <div
                        className="top3-bar-fill"
                        style={{
                          width: `${item.confidence * 100}%`,
                          backgroundColor: LABEL_COLORS[item.label] || '#2563eb',
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <Layers className="size-4 text-primary" />
                All Domain Scores
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="scores-grid">
                {LABELS.map((label) => {
                  const score = result.all_scores[label] || 0
                  const isActive = label === result.prediction
                  return (
                    <div
                      key={label}
                      className={`score-row ${isActive ? 'active' : ''}`}
                    >
                      <div className="score-label-container">
                        <div
                          className="score-dot"
                          style={{ backgroundColor: LABEL_COLORS[label] }}
                        />
                        <span className="score-label">{label}</span>
                      </div>
                      <div className="score-bar-bg">
                        <div
                          className="score-bar-fill"
                          style={{
                            width: `${Math.max(score * 100, 0.5)}%`,
                            backgroundColor: LABEL_COLORS[label],
                          }}
                        />
                      </div>
                      <span className="score-value">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function HowItWorksSection() {
  const steps = [
    {
      icon: Brain,
      title: 'Train Teacher Model',
      description: 'Fine-tune DistilBERT (67M params) on 65K IAB news articles across 11 domains. Achieves 80.7% macro-F1 on validation.',
    },
    {
      icon: Layers,
      title: 'Knowledge Distillation',
      description: 'Train a 4-layer, 256-hidden student (11.2M params) using teacher soft labels (KL divergence, T=3) + hard labels (CE). Alpha=0.7.',
    },
    {
      icon: Gauge,
      title: 'Deploy Efficient Model',
      description: 'Export 44 MB student model. Runs at 0.64 GFLOP (~6ms CPU latency) while retaining 95% of teacher accuracy (0.766 macro-F1).',
    },
  ]

  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">How It Works</h2>
        <p className="section-description">
          Three-step pipeline: large teacher → knowledge distillation → tiny deployable student
        </p>
      </div>
      <div className="content-grid">
        {steps.map((step, i) => (
          <div key={i} className="content-card">
            <div className="content-card-icon">
              <step.icon className="size-6" />
            </div>
            <h3 className="content-card-title">{step.title}</h3>
            <p className="content-card-text">{step.description}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function MetricsSection() {
  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">Model Performance</h2>
        <p className="section-description">Key metrics from test set evaluation (6,509 articles)</p>
      </div>
      <div className="metrics-grid">
        {METRICS.map((metric, i) => (
          <div key={i} className="metric-card">
            <div className="metric-value">{metric.value}</div>
            <div className="metric-label">{metric.label}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function LabelsSection() {
  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">11 Domain Labels</h2>
        <p className="section-description">IAB taxonomy — specific, mutually exclusive categories for news classification</p>
      </div>
      <div className="labels-table-wrapper">
        <table className="labels-table">
          <thead>
            <tr>
              <th>Label</th>
              <th>Description</th>
              <th>Example</th>
            </tr>
          </thead>
          <tbody>
            {LABELS.map((label) => (
              <tr key={label}>
                <td>
                  <span 
                    className="label-badge" 
                    style={{ 
                      backgroundColor: LABEL_COLORS[label] + '15',
                      borderColor: LABEL_COLORS[label] + '40',
                      color: LABEL_COLORS[label]
                    }}
                  >
                    <span 
                      className="label-color-dot"
                      style={{ backgroundColor: LABEL_COLORS[label] }}
                    />
                    {label}
                  </span>
                </td>
                <td>{LABEL_DESCRIPTIONS[label]}</td>
                <td>
                  <span style={{ color: '#64748b', fontSize: '0.875rem' }}>
                    {LABEL_EXAMPLES[label]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function PerClassPerformanceSection() {
  const sortedLabels = [...LABELS].sort((a, b) => PER_CLASS_F1[b] - PER_CLASS_F1[a])

  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">Per-Class Performance</h2>
        <p className="section-description">Macro F1 scores on test set — honest about strengths and weaknesses</p>
      </div>
      <div className="perf-grid">
        {sortedLabels.map((label) => {
          const f1 = PER_CLASS_F1[label]
          const strength = STRENGTH_LABELS[label]
          const color = f1 >= 0.8 ? '#22c55e' : f1 >= 0.7 ? '#3b82f6' : f1 >= 0.6 ? '#f59e0b' : '#ef4444'
          return (
            <div key={label} className="perf-item">
              <div className="perf-label" style={{ color: LABEL_COLORS[label] }}>
                <span 
                  style={{ 
                    display: 'inline-block', 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '3px', 
                    backgroundColor: LABEL_COLORS[label],
                    marginRight: '8px',
                    verticalAlign: 'middle'
                  }} 
                />
                {label}
              </div>
              <div className="perf-bar-bg">
                <div
                  className="perf-bar-fill"
                  style={{ width: `${f1 * 100}%`, backgroundColor: color }}
                />
              </div>
              <div className="perf-stats">
                <span>F1: <span className="f1-value">{f1.toFixed(3)}</span></span>
                <span className={`strength ${STRENGTH_CLASSES[strength]}`}>{strength}</span>
              </div>
            </div>
          )
        })}
      </div>
      <div className="card" style={{ marginTop: '1.5rem', background: '#fefce8', borderColor: '#fde047' }}>
        <CardContent className="py-4 px-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="size-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-amber-800 mb-1">Known Limitations</p>
              <p className="text-sm text-amber-700">
                <strong>Law & Legal (F1=0.537)</strong> and <strong>Education (F1=0.601)</strong> are the weakest classes — both have the smallest test sets (221 and 155 examples respectively). 
                The project includes 42K ambiguous rows in <code className="bg-amber-100 px-1 rounded">review_ambiguous.csv</code> that could be manually labeled to improve these classes.
              </p>
            </div>
          </div>
        </CardContent>
      </div>
    </section>
  )
}

function TechStackSection() {
  const techItems = [
    { icon: Brain, title: 'Teacher: DistilBERT', detail: '67M params, 6 layers, 768 hidden, fine-tuned on 52K articles' },
    { icon: Layers, title: 'Student: 4L-256H', detail: '11.2M params, 4 layers, 256 hidden, 4 attention heads' },
    { icon: Code, title: 'Knowledge Distillation', detail: 'KL divergence (T=3, α=0.7) + weighted cross-entropy, 2 seeds (42, 1337)' },
    { icon: Server, title: 'Backend: FastAPI', detail: 'Python API serving Hugging Face transformers model with URL fetching' },
    { icon: GitBranch, title: 'Frontend: React + Vite', detail: 'TypeScript, Tailwind CSS, shadcn/ui components' },
    { icon: BookOpen, title: 'Data: IAB News Dataset', detail: '106K articles from HuggingFace, filtered to 65K across 11 labels' },
  ]

  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">Tech Stack</h2>
        <p className="section-description">End-to-end pipeline from data to deployed model</p>
      </div>
      <div className="content-grid">
        {techItems.map((item, i) => (
          <div key={i} className="content-card">
            <div className="content-card-icon">
              <item.icon className="size-6" />
            </div>
            <h3 className="content-card-title">{item.title}</h3>
            <p className="content-card-text">{item.detail}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function RunLocallySection() {
  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">Run Locally</h2>
        <p className="section-description">Two ways to use the model</p>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <CardHeader>
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Server className="size-4" />
            Web Interface
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <pre className="code-block"><code>{`# Terminal 1: Start API server
python -m uvicorn app:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend && npm install && npm run dev

# Open http://localhost:5173`}</code></pre>
        </CardContent>
      </div>

      <div className="card">
        <CardHeader>
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Code className="size-4" />
            Python API
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <pre className="code-block"><code>{`from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "generated/deployed_student"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "Scientists discover high-efficiency solar cell material"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)
    pred = probs.argmax(dim=-1).item()
    print(f"Predicted: {model.config.id2label[pred]} ({probs[0][pred]:.1%})")`}</code></pre>
        </CardContent>
      </div>
    </section>
  )
}

function LimitationsSection() {
  const limitations = [
    {
      title: 'Weak Performance on Rare Classes',
      detail: 'Law & Legal (F1=0.54) and Education (F1=0.60) suffer from limited training data (221 and 155 test examples). The 42K ambiguous rows in review_ambiguous.csv could help if manually labeled.',
    },
    {
      title: 'Keyword-Based Environment Extraction',
      detail: 'Environment articles were filtered from Science using keywords (climate, pollution, renewable). This introduces noise — a dedicated Environment corpus would improve the 0.66 F1.',
    },
    {
      title: 'No Confidence Thresholding',
      detail: 'Model always predicts a class. Adding a reject option (e.g., max probability < 50%) would improve reliability for out-of-distribution inputs.',
    },
    {
      title: 'English-Only, News-Domain',
      detail: 'Trained on English news articles. Performance on other languages, domains (blogs, social media), or historical texts is untested.',
    },
    {
      title: 'Single-Label Classification',
      detail: 'Articles covering multiple topics (e.g., "Tech IPO raises $1B for climate startup") are forced into one label. Multi-label would be more realistic.',
    },
    {
      title: 'URL Fetching Limitations',
      detail: 'Article extraction works best on standard news sites. Paywalls, JavaScript-rendered content, and non-standard layouts may fail. RSS feeds with full content work well.',
    },
  ]

  return (
    <section className="content-section">
      <div className="section-header">
        <h2 className="section-title">Limitations & Future Work</h2>
        <p className="section-description">Transparent about where the model falls short and what could improve it</p>
      </div>
      <div className="content-grid">
        {limitations.map((item, i) => (
          <div key={i} className="content-card" style={{ borderLeft: '4px solid var(--color-danger)' }}>
            <div className="content-card-icon" style={{ background: 'var(--color-danger-light)', color: 'var(--color-danger)' }}>
              <AlertTriangle className="size-6" />
            </div>
            <h3 className="content-card-title">{item.title}</h3>
            <p className="content-card-text">{item.detail}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function FooterSection() {
  return (
    <footer className="footer">
      <div className="footer-links">
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
        <a href="#" target="_blank" rel="noopener noreferrer">Full Report</a>
        <a href="#" target="_blank" rel="noopener noreferrer">Model Card</a>
        <a href="#" target="_blank" rel="noopener noreferrer">Dataset (IAB)</a>
      </div>
      <p className="footer-note">
        Article Domain Tagger — Knowledge Distillation Project • 
        DistilBERT → Student 4L-256H • 11 IAB Domains • 0.64 GFLOP • Built with React, FastAPI & Hugging Face
      </p>
    </footer>
  )
}

function App() {
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [inputMode, setInputMode] = useState('text')

  const handlePredict = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult({ ...data, extracted_text: null })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFetchUrl = async () => {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/fetch-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!res.ok) {
        let errMsg = `Server error: ${res.status}`
        try {
          const err = await res.json()
          errMsg = err.detail || errMsg
        } catch {
          // Response wasn't JSON, use status text
          errMsg = res.statusText || errMsg
        }
        throw new Error(errMsg)
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <TooltipProvider>
      <div className="app-container">
        <HeroSection />
        
        <ClassifierSection 
          onClassify={handlePredict}
          onFetchUrl={handleFetchUrl}
          text={text}
          setText={setText}
          url={url}
          setUrl={setUrl}
          loading={loading}
          error={error}
          result={result}
          inputMode={inputMode}
          setInputMode={setInputMode}
        />

        <Separator className="divider" />

        <HowItWorksSection />
        <MetricsSection />
        <LabelsSection />
        <PerClassPerformanceSection />
        <TechStackSection />
        <RunLocallySection />
        <LimitationsSection />
        <FooterSection />
      </div>
    </TooltipProvider>
  )
}

export default App