import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  const features = [
    {
      title: "AI-Powered Analysis",
      description: "Advanced deep learning models trained on thousands of retinal images for accurate diabetic retinopathy detection."
    },
    {
      title: "Instant Results",
      description: "Get comprehensive analysis results in seconds, not hours. Fast and efficient screening process."
    },
    {
      title: "Medical Grade Accuracy",
      description: "Clinically validated models achieving 95%+ accuracy, suitable for medical screening applications."
    },
    {
      title: "Clinical Integration",
      description: "Seamlessly integrates with existing clinical workflows and electronic health record systems."
    }
  ];

  const stats = [
    { label: "Accuracy Rate", value: "95.8%" },
    { label: "Images Analyzed", value: "10K+" },
    { label: "Medical Centers", value: "25+" },
    { label: "Processing Time", value: "<5s" }
  ];

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              AI-Powered
              <span className="gradient-text"> Diabetic Retinopathy</span>
              <br />Detection & Classification
            </h1>
            <p className="hero-description">
              Advanced medical AI system that analyzes retinal fundus images to detect and classify 
              diabetic retinopathy with clinical-grade accuracy. Empowering healthcare professionals 
              with instant, reliable screening results.
            </p>

            <div className="hero-actions">
              <Link to="/upload" className="btn btn-primary">
                Start Analysis →
              </Link>
              <Link to="/about" className="btn btn-secondary">
                Learn More
              </Link>
            </div>

            <div className="hero-badges">
              <div className="badge">
                <span>✓ FDA Compliant</span>
              </div>
              <div className="badge">
                <span>🔒 HIPAA Secure</span>
              </div>
              <div className="badge">
                <span>📊 Clinically Validated</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="medical-card">
              <div className="card-header">
                <span>👁️ Retinal Analysis</span>
              </div>
              <div className="mock-image">
                <div className="retina-visualization">
                  <div className="optic-disc"></div>
                  <div className="blood-vessels"></div>
                  <div className="macula"></div>
                </div>
              </div>
              <div className="analysis-results">
                <div className="result-item">
                  <span className="label">Classification:</span>
                  <span className="value success">No DR Detected</span>
                </div>
                <div className="result-item">
                  <span className="label">Confidence:</span>
                  <span className="value">97.3%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stats-container">
          {stats.map((stat, index) => (
            <div key={stat.label} className="stat-card">
              <div className="stat-content">
                <div className="stat-value">{stat.value}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Advanced Medical AI Technology</h2>
            <p className="section-description">
              Our system combines cutting-edge computer vision with medical expertise to provide 
              accurate, reliable diabetic retinopathy screening.
            </p>
          </div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={feature.title} className="feature-card">
                <div className="feature-icon">
                  🧠
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2 className="cta-title">Ready to Start Screening?</h2>
          <p className="cta-description">
            Upload a retinal fundus image and get instant AI-powered analysis results 
            with clinical recommendations.
          </p>
          <Link to="/upload" className="btn btn-primary btn-large">
            📤 Upload Image Now →
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;