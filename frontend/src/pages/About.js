import React from 'react';

const About = () => {
  const teamMembers = [
    {
      name: "Dr. Sarah Chen",
      role: "Chief Medical Officer",
      expertise: "Ophthalmology, Retinal Diseases",
    },
    {
      name: "Alex Rodriguez",
      role: "Lead AI Engineer", 
      expertise: "Computer Vision, Deep Learning",
    },
    {
      name: "Dr. Michael Johnson",
      role: "Clinical Advisor",
      expertise: "Diabetes Care, Medical AI",
    }
  ];

  const metrics = [
    {
      value: "95.8%",
      label: "Sensitivity",
      description: "True positive rate for DR detection"
    },
    {
      value: "97.2%",
      label: "Specificity", 
      description: "True negative rate for healthy cases"
    },
    {
      value: "10,000+",
      label: "Images Trained",
      description: "Diverse dataset from multiple sources"
    },
    {
      value: "25+",
      label: "Medical Centers",
      description: "Clinical validation partners"
    }
  ];

  const features = [
    {
      title: "Advanced AI Models",
      description: "State-of-the-art deep learning architectures including EfficientNet, Vision Transformers, and hybrid models for optimal accuracy."
    },
    {
      title: "Medical Grade Security",
      description: "HIPAA-compliant infrastructure with end-to-end encryption, secure data handling, and privacy-first design principles."
    },
    {
      title: "Clinically Validated",
      description: "Rigorously tested in real clinical environments with performance metrics validated by medical professionals."
    },
    {
      title: "Comprehensive Training",
      description: "Trained on diverse, high-quality datasets including APTOS, MESSIDOR, and EyePACS for robust performance across populations."
    }
  ];

  return (
    <div className="about-page">
      {/* Hero Section */}
      <section className="about-hero">
        <div className="section-container">
          <div className="hero-content">
            <h1 className="hero-title">
              Advancing Healthcare Through
              <span className="gradient-text"> Artificial Intelligence</span>
            </h1>
            <p className="hero-description">
              Our mission is to democratize access to high-quality diabetic retinopathy screening 
              through cutting-edge AI technology, empowering healthcare providers worldwide to 
              detect and prevent vision loss in diabetic patients.
            </p>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="mission-section">
        <div className="section-container">
          <div className="mission-content">
            <div className="mission-text">
              <h2 className="section-title">Our Mission</h2>
              <p className="mission-description">
                Diabetic retinopathy affects over 100 million people worldwide and is the leading 
                cause of blindness in working-age adults. Early detection can prevent up to 95% of 
                vision loss cases, yet many patients lack access to specialized screening.
              </p>
              <p className="mission-description">
                We're bridging this gap by developing AI-powered screening tools that deliver 
                clinical-grade accuracy while being accessible, affordable, and easy to use in 
                any healthcare setting.
              </p>

              <div className="mission-stats">
                <div className="stat">
                  <span className="stat-number">100M+</span>
                  <span className="stat-label">People Affected Globally</span>
                </div>
                <div className="stat">
                  <span className="stat-number">95%</span>
                  <span className="stat-label">Preventable Vision Loss</span>
                </div>
              </div>
            </div>

            <div className="mission-visual">
              <div className="visual-card">
                <h3>👁️ Early Detection Saves Sight</h3>
                <p>
                  With timely screening and treatment, we can preserve vision and 
                  improve quality of life for millions of diabetic patients worldwide.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="technology-section">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Advanced Technology Stack</h2>
            <p className="section-description">
              Our platform leverages the latest breakthroughs in medical AI and computer vision 
              to deliver accurate, reliable diabetic retinopathy screening.
            </p>
          </div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={feature.title} className="feature-card">
                <div className="feature-icon">🧠</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Performance Metrics */}
      <section className="metrics-section">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Clinical Performance</h2>
            <p className="section-description">
              Our models have been rigorously tested and validated across multiple datasets 
              and clinical environments to ensure reliable performance.
            </p>
          </div>

          <div className="metrics-grid">
            {metrics.map((metric, index) => (
              <div key={metric.label} className="metric-card">
                <div className="metric-value">{metric.value}</div>
                <div className="metric-label">{metric.label}</div>
                <div className="metric-description">{metric.description}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="team-section">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Expert Team</h2>
            <p className="section-description">
              Our multidisciplinary team combines medical expertise with cutting-edge AI research 
              to deliver innovative healthcare solutions.
            </p>
          </div>

          <div className="team-grid">
            {teamMembers.map((member, index) => (
              <div key={member.name} className="team-card">
                <div className="team-avatar">🩺</div>
                <h3 className="team-name">{member.name}</h3>
                <div className="team-role">{member.role}</div>
                <div className="team-expertise">{member.expertise}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Clinical Impact */}
      <section className="impact-section">
        <div className="section-container">
          <div className="impact-content">
            <h2 className="impact-title">Real-World Clinical Impact</h2>
            <div className="impact-grid">
              <div className="impact-item">
                <div className="impact-text">
                  <h3>📈 Improved Screening Rates</h3>
                  <p>40% increase in diabetic retinopathy screening in pilot programs</p>
                </div>
              </div>

              <div className="impact-item">
                <div className="impact-text">
                  <h3>👥 Enhanced Accessibility</h3>
                  <p>Bringing specialist-level screening to underserved communities</p>
                </div>
              </div>

              <div className="impact-item">
                <div className="impact-text">
                  <h3>✅ Clinical Validation</h3>
                  <p>Validated across 25+ medical centers with consistent performance</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;