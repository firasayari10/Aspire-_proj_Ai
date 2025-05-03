import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import HomePage from './components/HomePage';
import ImageDropzone from './components/ImageDropzone';
import BackButton from './components/BackButton';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/images" element={
            <>
              <BackButton />
              <main className="App-main">
                <ImageDropzone />
              </main>
            </>
          } />
          <Route path="/drone" element={
            <>
              <BackButton />
              <div className="coming-soon">
                <h2>Drone Footage Analysis</h2>
                <p>This feature is coming soon!</p>
              </div>
            </>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 