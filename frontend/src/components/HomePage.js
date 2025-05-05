import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';
import espritLogo from '../images/esprit.png';
import secondlifelogo from '../images/Logo Second Life final.png';
import DroneImageDropzone from './DroneImageDropzone';

const HomePage = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [currentReview, setCurrentReview] = useState(0);
  const navigate = useNavigate();

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };


  const ArrowRightIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12"></line>
      <polyline points="12 5 19 12 12 19"></polyline>
    </svg>
  );

  const ChevronLeftIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6"></polyline>
    </svg>
  );

  const ChevronRightIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
  );

  const StarIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#FFD700" stroke="#FFD700" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
    </svg>
  );

  const TargetIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <circle cx="12" cy="12" r="6"></circle>
      <circle cx="12" cy="12" r="2"></circle>
    </svg>
  );

 
  const reviews = [
    {
      id: 1,
      name: "John Smith",
      role: "City Waste Management",
      content: "This waste detection system has revolutionized how we identify and manage waste in urban areas. The accuracy is remarkable!",
      rating: 5,
    },
    {
      id: 2,
      name: "Sarah Johnson",
      role: "Environmental Scientist",
      content: "As a researcher, I've found this tool invaluable for tracking waste patterns and improving our community cleanup initiatives.",
      rating: 5,
    },
    {
      id: 3,
      name: "Michael Chen",
      role: "Recycling Plant Manager",
      content: "The drone footage analysis has helped us optimize our recycling processes and increase efficiency by 30%.",
      rating: 4,
    },
  ];


const partners = [
  { name: "Esprit", logo: espritLogo },
  { name: "Second Life", logo: secondlifelogo }
];
   
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentReview((prev) => (prev === reviews.length - 1 ? 0 : prev + 1));
    }, 5000);
    return () => clearInterval(interval);
  }, [reviews.length]);

  const nextReview = () => {
    setCurrentReview((prev) => (prev === reviews.length - 1 ? 0 : prev + 1));
  };

  const prevReview = () => {
    setCurrentReview((prev) => (prev === 0 ? reviews.length - 1 : prev - 1));
  };

  return (
    <div className="home-container">
      <button className="menu-toggle" onClick={toggleMenu}>
        {menuOpen ? 'Close' : 'Menu'}
      </button>
      <div className={`side-menu ${menuOpen ? 'open' : ''}`}>
        <ul className="menu-items">
          <li onClick={() => navigate('/')}>Home</li>
          <li onClick={() => navigate('/drone')}>Drone Footage</li>
          <li onClick={() => navigate('/images')}>Normal Images</li>
          <li onClick={() => navigate('/about')}>About</li>
          <li onClick={() => navigate('/contact')}>Contact Us</li>
        </ul>
      </div>

      {/* Hero Banner */}
      <div className="hero-banner">
        <div className="banner-content">
          <h1>DispoEasy</h1>
          <p>Using AI and drone technology to create cleaner communities</p>
          <button className="cta-button" onClick={() => navigate('/drone')}>
            Get Started <ArrowRightIcon />
          </button>
        </div>
      </div>

      <div className="content">
        <h1>Welcome to the Waste Detection System</h1>
        <p>Analyze waste using drone footage or regular images.</p>
        <div className="options-container">
          <div 
            className="option-card"
            onClick={() => navigate('/drone')}
          >
            <div className="option-icon">🛸</div>
            <h2>Drone Footage</h2>
            <p>Process and analyze waste from drone captured videos</p>
          </div>

          <div 
            className="option-card"
            onClick={() => navigate('/images')}
          >
            <div className="option-icon">📸</div>
            <h2>Normal Images</h2>
            <p>Upload and analyze waste from regular images</p>
          </div>
        </div>
        
        {/* Mission Statement Section */}
        <section className="mission-section">
          <div className="mission-icon">
            <TargetIcon />
          </div>
          <h2>Our Mission</h2>
          <p>
            We're committed to creating a cleaner world through innovative technology. 
            Our waste detection system empowers communities, governments, and businesses 
            to identify, track, and manage waste more efficiently than ever before.
          </p>
          <div className="stats-container">
            <div className="stat-item">
              <h3>95%</h3>
              <p>Detection Accuracy</p>
            </div>
            <div className="stat-item">
              <h3>30+</h3>
              <p>Cities Served</p>
            </div>
            <div className="stat-item">
              <h3>1M+</h3>
              <p>Images Analyzed</p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <h2>Why Choose Our Solution</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h3>AI-Powered Analysis</h3>
              <p>Advanced algorithms identify and classify waste with high accuracy</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-time Processing</h3>
              <p>Get results quickly to make timely waste management decisions</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>Detailed Reports</h3>
              <p>Comprehensive insights with actionable data visualization</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔄</div>
              <h3>Versatile Input</h3>
              <p>Works with both drone footage and standard images</p>
            </div>
          </div>
        </section>

        {/* Reviews Section */}
        <section className="reviews-section">
          <h2>What Our Users Say</h2>
          <div className="reviews-carousel">
            <button className="carousel-button prev" onClick={prevReview}>
              <ChevronLeftIcon />
            </button>
            <div className="review-card">
              <div className="review-stars">
                {[...Array(reviews[currentReview].rating)].map((_, i) => (
                  <StarIcon key={i} />
                ))}
              </div>
              <p className="review-content">"{reviews[currentReview].content}"</p>
              <div className="reviewer-info">
                <p className="reviewer-name">{reviews[currentReview].name}</p>
                <p className="reviewer-role">{reviews[currentReview].role}</p>
              </div>
            </div>
            <button className="carousel-button next" onClick={nextReview}>
              <ChevronRightIcon />
            </button>
          </div>
          <div className="review-indicators">
            {reviews.map((_, index) => (
              <span 
                key={index} 
                className={`indicator ${index === currentReview ? 'active' : ''}`}
                onClick={() => setCurrentReview(index)}
              />
            ))}
          </div>
        </section>

        {/* Partners Section */}
        <section className="partners-section">
  <h2>Our Partners</h2>
  <div className="partners-container">
    {partners.map((partner, index) => (
      <div key={index} className="partner-card">
        <img 
          src={partner.logo} 
          alt={partner.name} 
          className="partner-logo" 
        />
        <p className="partner-name">{partner.name}</p>
      </div>
    ))}
  </div>
</section>



        {/* Call to Action */}
        <section className="cta-section">
          <div className="cta-content">
            <h2>Ready to Transform Waste Management?</h2>
            <p>Join organizations worldwide using our technology to create cleaner communities.</p>
            <div className="cta-buttons">
              <button className="primary-button" onClick={() => navigate('/drone')}>Try Drone Analysis</button>
              <button className="secondary-button" onClick={() => navigate('/contact')}>Contact Sales</button>
            </div>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="site-footer">
        <div className="footer-columns">
          <div className="footer-column">
            <h3>Waste Detection System</h3>
            <p>Innovative AI-powered waste detection and management solutions.</p>
          </div>
          <div className="footer-column">
            <h3>Quick Links</h3>
            <ul>
              <li onClick={() => navigate('/')}>Home</li>
              <li onClick={() => navigate('/drone')}>Drone Analysis</li>
              <li onClick={() => navigate('/images')}>Image Analysis</li>
              <li onClick={() => navigate('/about')}>About Us</li>
            </ul>
          </div>
          <div className="footer-column">
            <h3>Contact</h3>
            <p>Email: info@wastedetection.com</p>
            <p>Phone: (555) 123-4567</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Waste Detection System. All Rights Reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;