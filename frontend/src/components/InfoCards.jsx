import React from 'react';

function InfoCards({ darkMode }) {
  const cards = [
    {
      title: "Pneumonia Facts",
      icon: "📊",
      items: [
        "Pneumonia causes 15% of all deaths in children under 5 worldwide",
        "Vaccines can prevent some types of pneumonia",
        "Symptoms include cough, fever, and difficulty breathing"
      ]
    },
    {
      title: "Prevention Tips",
      icon: "🛡️",
      items: [
        "Get vaccinated against flu and pneumococcal disease",
        "Wash hands regularly with soap and water",
        "Avoid smoking and secondhand smoke"
      ]
    },
    {
      title: "When to Seek Help",
      icon: "⚠️",
      items: [
        "Difficulty breathing or shortness of breath",
        "Chest pain that worsens when breathing",
        "High fever (above 102°F or 39°C) that persists"
      ]
    }
  ];

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Learn About Pneumonia</h2>
      <div className="space-y-6">
        {cards.map((card, index) => (
          <div 
            key={index} 
            className={`p-5 rounded-xl border ${
              darkMode ? 'bg-gray-800/30 border-gray-700' : 'bg-white border-gray-200'
            } shadow-sm`}
          >
            <div className="flex items-center mb-3">
              <span className="text-2xl mr-3">{card.icon}</span>
              <h3 className="text-lg font-bold">{card.title}</h3>
            </div>
            <ul className="space-y-2">
              {card.items.map((item, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mt-2 mr-3"></span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
        
        <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 text-white">
          <h3 className="font-bold text-lg mb-2">Take Our Health Quiz</h3>
          <p className="mb-4">Test your knowledge about pneumonia prevention and symptoms</p>
          <button className="px-4 py-2 bg-white text-blue-600 rounded-full font-medium">
            Start Quiz
          </button>
        </div>
      </div>
    </div>
  );
}

export default InfoCards;