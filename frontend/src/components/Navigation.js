import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Upload', href: '/upload' },
    { name: 'Results', href: '/results' },
    { name: 'About', href: '/about' }
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="medical-nav">
      <div className="nav-container">
        {/* Logo */}
        <Link to="/" className="nav-logo">
          <div className="logo-icon">
            🔬
          </div>
          <div className="logo-text">
            <span className="logo-main">DR Classifier</span>
            <span className="logo-sub">Medical AI</span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="nav-links desktop-only">
          {navigation.map((item) => {
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`nav-link ${isActive(item.href) ? 'active' : ''}`}
              >
                <span>{item.name}</span>
                {isActive(item.href) && (
                  <div className="nav-indicator" />
                )}
              </Link>
            );
          })}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="mobile-menu-button mobile-only"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          {isMobileMenuOpen ? '✕' : '☰'}
        </button>
      </div>

      {/* Mobile Navigation Menu */}
      {isMobileMenuOpen && (
        <div className="mobile-nav mobile-only">
          {navigation.map((item) => {
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`mobile-nav-link ${isActive(item.href) ? 'active' : ''}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <span>{item.name}</span>
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
};

export default Navigation;