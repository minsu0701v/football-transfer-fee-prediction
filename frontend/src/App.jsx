import { useState } from 'react'
import './App.css'

function App() {

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [selectedPlayer, setSelectedPlayer] = useState(null)
  const [destinationLeague, setDestinationLeague] = useState('GB1')
  const [prediction, setPrediction] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')

  const searchPlayers = async () => {
    const query = searchQuery.trim()

    if (!query) {
      return
    }

    try {
      setSearchLoading(true)
      setSearchError('')

      const response = await fetch(
        `http://127.0.0.1:8000/players/search?q=${encodeURIComponent(query)}&limit=8`
      )

      if (!response.ok) {
        throw new Error('선수 검색에 실패했습니다.')
      }

      const data = await response.json()

      setSearchResults(data)
    } catch (error) {
      console.error(error)
      setSearchError('선수를 불러오지 못했습니다.')
      setSearchResults([])
    } finally {
      setSearchLoading(false)
    }
  }

  const selectPlayer = (player) => {
    setSelectedPlayer(player)
    setSearchQuery(player.player_name)
    setSearchResults([])
  }

  const predictTransferFee = async () => {
    if (!selectedPlayer) {
      setPredictionError('먼저 선수를 선택해주세요.')
      return
    }

    try {
      setPredictionLoading(true)
      setPredictionError('')

      const response = await fetch(
        'http://127.0.0.1:8000/predict',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            player_id: selectedPlayer.player_id,
            to_league_id: destinationLeague,
          }),
        }
      )

      if (!response.ok) {
        throw new Error('이적료 예측에 실패했습니다.')
      }

      const data = await response.json()
      setPrediction(data)

    } catch (error) {
      console.error(error)
      setPredictionError('이적료를 예측하지 못했습니다.')
    } finally {
      setPredictionLoading(false)
    }
  }

  const shapItems = prediction?.explanation ?? [];

  const maxShapImpact =
    shapItems.length > 0
      ? Math.max(...shapItems.map((item) => Math.abs(item.impact)))
      : 1;

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>⚽ Football</h2>
        <h2>Transfer Predictor</h2>

        <nav>
          <p>⌂ 홈</p>
          <p>□ 예측 기록</p>
          <p>☆ 관심 선수</p>
          <p>▥ 통계 대시보드</p>
          <p>ⓘ 모델 설명</p>
          <p>♙ 팀</p>
        </nav>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="search-area">
            <div className="search-box">
              <span className="search-icon">⌕</span>

              <input
                type="text"
                placeholder="선수 이름을 입력하세요 (예: Son Heung-min)"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    searchPlayers()
                  }
                }}
              />

              <span className="shortcut">Enter</span>
            </div>

            {(searchLoading || searchError || searchResults.length > 0) && (
              <div className="search-dropdown">
                {searchLoading && (
                  <p className="search-message">검색 중...</p>
                )}

                {searchError && (
                  <p className="search-message error">{searchError}</p>
                )}

                {!searchLoading &&
                  searchResults.map((player) => (
                    <button
                      key={player.player_id}
                      className="search-result-item"
                      onClick={() => selectPlayer(player)}
                    >
                      {player.player_image_url && (
                        <img
                          src={player.player_image_url}
                          alt={player.player_name}
                        />
                      )}

                      <div>
                        <strong>{player.player_name}</strong>
                        <span>
                          {player.current_club_name} · {player.current_league_name}
                        </span>
                      </div>
                    </button>
                  ))}
              </div>
            )}
          </div>

          <div className="top-actions">
            <span className="help">?</span>
            <span className="user">● 사용자⌄</span>
          </div>
        </header>

        <section className="dashboard-top">
          <div className="card player-card">
            <h2>선수 프로필</h2>

            <div className="player-info">
              <div className="player-image">
                {selectedPlayer?.player_image_url ? (
                  <img
                    src={selectedPlayer.player_image_url}
                    alt={selectedPlayer.player_name}
                  />
                ) : (
                  <span>선수 사진</span>
                )}
              </div>

              <div className="player-details">
                <h2>
                  {selectedPlayer?.player_name ?? '선수를 검색하세요'}
                </h2>

                <p>
                  리그　{selectedPlayer?.current_league_name ?? '-'}
                </p>

                <p>
                  포지션　{selectedPlayer?.main_position ?? '-'}
                </p>

                <p>
                  현재 팀　{selectedPlayer?.current_club_name ?? '-'}
                </p>
              </div>
            </div>
          </div>

          <div className="card season-card">
            <h2>이번 시즌 기록</h2>

            <div className="season-stats">
              <div>
                <span>경기</span>
                <strong>{selectedPlayer?.matches ?? '-'}</strong>
              </div>

              <div>
                <span>선발</span>
                <strong>{selectedPlayer?.started ?? '-'}</strong>
              </div>

              <div>
                <span>출전 시간</span>
                <strong>{selectedPlayer?.minutes?.toLocaleString() ?? '-'}</strong>
              </div>

              <div>
                <span>평점</span>
                <strong className="highlight">{selectedPlayer?.rating ?? '-'}</strong>
              </div>

              <div>
                <span>골</span>
                <strong>{selectedPlayer?.goals ?? '-'}</strong>
              </div>

              <div>
                <span>도움</span>
                <strong>{selectedPlayer?.assists ?? '-'}</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="dashboard-bottom">
          <div className="card prediction-form">
            <h2>예측 조건 설정</h2>

            <div className="form-group">
              <label>예상 이적 리그</label>

              <select
                value={destinationLeague}
                onChange={(event) => setDestinationLeague(event.target.value)}
              >
                <option value="GB1">Premier League</option>
                <option value="ES1">La Liga</option>
                <option value="L1">Bundesliga</option>
                <option value="IT1">Serie A</option>
                <option value="FR1">Ligue 1</option>
              </select>
            </div>

            <button
              className="predict-button"
              onClick={predictTransferFee}
              disabled={predictionLoading}
            >
              {predictionLoading
                ? '예측 중...'
                : '이적료 예측하기'}
            </button>
          </div>

          <div className="card prediction-result">
            <h2>예측 결과</h2>

            {predictionError && (
              <p className="prediction-error">
                {predictionError}
              </p>
            )}

            {prediction ? (
              <div className="result-content">
                <p className="result-label">예상 이적료</p>

                <strong className="result-fee">
                  €{prediction.predicted_transfer_fee_million.toFixed(1)}M
                </strong>

                <div className="prediction-summary">
                  <span>{prediction.player_name}</span>
                  <span>→</span>
                  <span>
                    {
                      {
                        GB1: 'Premier League',
                        ES1: 'La Liga',
                        L1: 'Bundesliga',
                        IT1: 'Serie A',
                        FR1: 'Ligue 1',
                      }[prediction.to_league_id]
                    }
                  </span>
                </div>
              </div>
            ) : (
              <div className="result-empty">
                <p>선수와 목적 리그를 선택한 후</p>
                <p>이적료를 예측해보세요.</p>
              </div>
            )}
          </div>

          <div className="card shap-card">
            <h2>예측 설명 (SHAP)</h2>

            <p className="shap-description">
              이적료 예측에 가장 큰 영향을 준 요인입니다.
            </p>

            {shapItems.length > 0 ? (
              <>
                <div className="shap-list">
                  {shapItems.slice(0, 6).map((item) => {
                    const isPositive = item.direction === "increase";

                    const width =
                      (Math.abs(item.impact) / maxShapImpact) * 100;

                    return (
                      <div
                        className="shap-row"
                        key={item.feature_name}
                      >
                        <span className="shap-name">
                          {item.feature}
                        </span>

                        <div className="shap-bar-area">
                          <div
                            className={`shap-bar ${isPositive ? "positive" : "negative"
                              }`}
                            style={{
                              width: `${width}%`,
                            }}
                          />
                        </div>

                        <span
                          className={`shap-value ${isPositive
                            ? "positive-text"
                            : "negative-text"
                            }`}
                        >
                          {isPositive ? "+" : "-"}
                          {Math.abs(item.impact).toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="shap-legend">
                  <span className="positive-text">
                    ▲ 이적료 상승
                  </span>

                  <span className="negative-text">
                    ▼ 이적료 하락
                  </span>
                </div>
              </>
            ) : (
              <p className="shap-description">
                이적료를 예측하면 SHAP 분석 결과가 표시됩니다.
              </p>
            )}
          </div>
        </section>

      </main>
    </div>
  )
}

export default App