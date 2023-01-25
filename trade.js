const alpaca = require('@alpacahq/alpaca-trade-api');
const tf = require('@tensorflow/tfjs');
const ppo = require('@tensorflow/tfjs-agents/dist/ppo');


var api_key = ''
var secret_key = ''
var base_url= 'https://paper-api.alpaca.markets'

// Instantiate the Alpaca API client
const api = new alpaca({
    keyId: api_key,
    secretKey: secret_key,
    paper: true,
    usePolygon: false
});

// Create the PPO agent
const agent = new ppo.PPOAgent({
    stateSize: 3,
    actionSize: 2,
    hiddenUnits: [64, 64],
    optimizer: tf.train.adam()
  });

// Train the agent using historical data
// Set up a function to get real-time stock data
async function getData() {
    try {
        console.log('getting data');
        const barset = await api.get_barset('AAPL', '1Min', { limit: 100 });
        const data = barset['AAPL'];
        const preprocessedData = preprocess(data);
        // Train the agent
        for (let i = 0; i < preprocessedData.length; i++) {
            const state = tf.tensor2d(preprocessedData[i].state, [1, 5]);
            const action = tf.tensor1d(preprocessedData[i].action, 'int32');
            const reward = preprocessedData[i].reward;
            const nextState = tf.tensor2d(preprocessedData[i].nextState, [1, 5]);
            const done = preprocessedData[i].done;
            agent.train(state, action, reward, nextState, done);
        }
        console.log('training finished');
        return true;
    } catch (err) {
        console.error(err);
        return false;
    }
}

// Set up a function to make trades
async function makeTrade(state) {
    // Use the agent to predict the next action
    const trainAgent = await getData();
    if (trainAgent) {
        console.log('started predicting');
        const action = agent.predict(state);

        // Use the Alpaca API to place an order based on the action
        try {
            console.log('started order');
            const order = await api.createOrder({
                symbol: 'AAPL',
                qty: action.qty,
                side: action.side,
                type: 'market',
                time_in_force: 'gtc'
            });
            console.log(`Order placed: ${order.id}`);
        } catch (err) {
            console.error(err);
        }
    } else {
        console.log('Failure');
    }
}

async function getCurrentMarketState() {
    try {
      // Get the current market state
      const ticker = await api.get_ticker('AAPL');
  
      // Calculate the Sharpe ratio
      const returns = await api.get_returns('AAPL', 'day', 1);
      const sharpeRatio = (returns.mean - 0.02) / returns.stddev;
  
      // Preprocess and normalize the data
      const state = {
        stockPrice: ticker.last.toFixed(2),
        sharpeRatio: sharpeRatio.toFixed(4),
        volume: ticker.volume
      };
      // Normalizing the data
      state.stockPrice = (state.stockPrice - MIN_PRICE) / (MAX_PRICE - MIN_PRICE);
      state.sharpeRatio = (state.sharpeRatio - MIN_SHARPE) / (MAX_SHARPE - MIN_SHARPE);
      state.volume = (state.volume - MIN_VOLUME) / (MAX_VOLUME - MIN_VOLUME);
      return state;
    } catch (err) {
      console.error(err);
      return null;
    }
  }

// Set up a function to check the account balance
async function checkBalance() {
    try {
        const account = await api.getAccount();
        console.log(`Current balance: ${account.cash}`);
    } catch (err) {
        console.error(err); 
    }
}



async function initiateTrade () {
  const currentState = await getCurrentMarketState();
  makeTrade(currentState);
  //logging balance after trade
  await checkBalance()
}

initiateTrade();

