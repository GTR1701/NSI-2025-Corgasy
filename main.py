import sys
from src.tournament import Tournament
from base_bot import BaseBot


def load_participant_bot():
    """
    Ładuje bota uczestnika z pliku solution.py.
    """
    try:
        from solution import create_bot

        bot_instance = create_bot()

        # Walidacja, czy bot dziedziczy po klasie BaseBot
        if not isinstance(bot_instance, BaseBot):
            raise TypeError(
                f"Bot musi dziedziczyć po klasie BaseBot!"
                f"Otrzymano: {type(bot_instance)}"
            )

        print(f"Bot załadowany: {bot_instance.__class__.__name__}")
        return bot_instance
    except ImportError as e:
        print("BŁĄD: Nie znaleziono funkcji 'create_bot()' w solution.py")
        print(f"Szczegóły: {e}")
        print("Utwórz funkcję create_bot() która zwraca instancję Twojej klasy bota.")
        sys.exit(1)


def load_calculate_reward():
    """
    Ładuje funkcję calculate_reward z pliku solution.py.
    """
    try:
        from solution import calculate_reward

        if not callable(calculate_reward):
            raise TypeError("calculate_reward musi być funkcją.")

        print("Funkcja calculate_reward załadowana.")
        return calculate_reward
    except ImportError as e:
        print("BŁĄD: Nie znaleziono funkcji 'calculate_reward()' w solution.py")
        print(f"Szczegóły: {e}")
        print(
            "Utwórz funkcję calculate_reward(game_state: dict) która zwraca wartość nagrody (float)."
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        participant_bot = load_participant_bot()
        try:
            from solution import train_bot
            train_bot(episodes=5000)
        except ImportError:
            def train_bot(bot: BaseBot, episodes: int):
                """Domyślna funkcja treningowa, jeśli nie jest zdefiniowana w solution.py."""
                print("Funkcja train_bot() nie została zdefiniowana w solution.py.")
                print("Używana jest domyślna funkcja treningowa (brak treningu).")
                return
        # Załaduj bota uczestnika

        # Wczytaj opcjonalne ustawienia turnieju
        try:
            from solution import BENCHMARK_EPISODES
        except ImportError:
            BENCHMARK_EPISODES = 50
        try:
            from solution import WATCH_GAME
        except ImportError:
            WATCH_GAME = False
        # Uruchom turniej
        calculate_reward = load_calculate_reward()
        tournament = Tournament(calculate_reward, participant_bot)
        tournament.run_benchmark(episodes=BENCHMARK_EPISODES)

        if WATCH_GAME:
            tournament.watch_game()

    except Exception as e:
        print(f"BŁĄD: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
