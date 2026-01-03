import pickle
import numpy as np
import subprocess
import os
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Genre definitions

GENRES = {
    'classical': {
        'name': 'Classical',
        'tempo': 120,
        'scale': [60, 62, 64, 65, 67, 69, 71, 72],  # C Major
        'chord_type': 'triad',
        'program': {'melody': 0, 'chords': 0, 'bass': 32},  # Piano + Acoustic Bass
        'melody_range': (60, 84),
        'melody_steps': [-2, -1, 0, 1, 2],
        'step_probs': [0.20, 0.30, 0.10, 0.30, 0.10],
    },
    'jazz': {
        'name': 'Jazz',
        'tempo': 140,
        'scale': [60, 62, 63, 65, 67, 68, 70, 72],  # C Blues scale
        'chord_type': 'seventh',
        'program': {'melody': 11, 'chords': 0, 'bass': 32},  # Vibraphone + Piano + Bass
        'melody_range': (55, 80),
        'melody_steps': [-4, -2, -1, 0, 1, 2, 4],
        'step_probs': [0.10, 0.15, 0.20, 0.10, 0.20, 0.15, 0.10],
    },
    'rock': {
        'name': 'Rock',
        'tempo': 130,
        'scale': [60, 62, 63, 65, 67, 69, 70, 72],  # C Minor Pentatonic
        'chord_type': 'triad',
        'program': {'melody': 29, 'chords': 30, 'bass': 33},  # Overdriven + Distortion + Electric Bass
        'melody_range': (55, 76),
        'melody_steps': [-5, -3, -1, 0, 1, 3, 5],
        'step_probs': [0.10, 0.15, 0.20, 0.10, 0.20, 0.15, 0.10],
    },
    'electronic': {
        'name': 'Electronic',
        'tempo': 128,
        'scale': [60, 62, 64, 67, 69, 72],  # Pentatonic
        'chord_type': 'triad',
        'program': {'melody': 81, 'chords': 88, 'bass': 38},  # Lead + Synth Pad + Synth Bass
        'melody_range': (60, 84),
        'melody_steps': [-4, -2, 0, 2, 4],
        'step_probs': [0.20, 0.25, 0.10, 0.25, 0.20],
    },
    'blues': {
        'name': 'Blues',
        'tempo': 100,
        'scale': [60, 63, 65, 66, 67, 70, 72],  # C Blues
        'chord_type': 'seventh',
        'program': {'melody': 25, 'chords': 0, 'bass': 32},  # Steel Guitar + Piano + Bass
        'melody_range': (55, 76),
        'melody_steps': [-3, -2, -1, 0, 1, 2, 3],
        'step_probs': [0.10, 0.15, 0.25, 0.10, 0.25, 0.15, 0.10],
    },
}

def load_models(model_path='models.pkl'):
    """Load ML models"""
    try:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        return models['rf_velocity'], models['rf_duration'], models['scaler']
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def predict_ml_params(notes, rf_vel, rf_dur, scaler):
    """Predict velocity and duration with ML for a sequence of notes"""
    if len(notes) < 2 or rf_vel is None:
        # Fallback: Musical dynamic curve
        velocities = []
        for i in range(len(notes)):
            # Create natural dynamics: crescendo and diminuendo
            pos = i / max(len(notes) - 1, 1)
            # Sinusoidal dynamics for musicality
            dynamic = 70 + 40 * np.sin(pos * 2 * np.pi)
            velocities.append(int(dynamic))
        return velocities, [600] * len(notes)
    
    X = []
    for i, pitch in enumerate(notes):
        prev_p = notes[i-1] if i > 0 else pitch
        next_p = notes[i+1] if i < len(notes)-1 else pitch
        
        pitch_interval = abs(pitch - prev_p)
        seq_pos = i / len(notes)
        local_density = min(12, sum(1 for j in range(max(0, i-6), min(len(notes), i+6))))
        duration = 480  # Default duration for prediction
        
        # 7 features
        feat = [pitch, pitch_interval, prev_p, next_p, seq_pos, local_density, duration]
        X.append(feat)
    
    X = np.array(X)
    X_scaled = scaler.transform(X)
    
    pred_vel = rf_vel.predict(X_scaled)
    pred_dur = rf_dur.predict(X_scaled)
    
    # Ensure velocities are in valid MIDI range
    pred_vel = np.clip(np.abs(pred_vel), 1, 127).astype(int)
    
    # Ensure note durations are reasonable
    pred_dur = np.clip(np.abs(pred_dur), 200, 1920).astype(int)
    
    # Convert to Python int (not numpy int64) for MIDI library compatibility
    return [int(v) for v in pred_vel], [int(d) for d in pred_dur]

def compose_melody(scale, num_notes=16, start_pitch=60, melody_steps=None, step_probs=None, melody_range=(36, 84)):
    """Generate a melody in a given scale with controlled leaps"""
    melody = []
    current = start_pitch
    
    if melody_steps is None:
        melody_steps = [-3, -2, -1, 0, 1, 2, 3]
    if step_probs is None:
        step_probs = [0.05, 0.15, 0.25, 0.1, 0.25, 0.15, 0.05]
    
    # Normalize probabilities to ensure they sum to 1.0
    step_probs = np.array(step_probs)
    step_probs = step_probs / step_probs.sum()
    
    for _ in range(num_notes):
        # Melodic movement according to style
        step = np.random.choice(melody_steps, p=step_probs)
        current += step
        current = np.clip(current, melody_range[0], melody_range[1])
        
        # Quantize to scale
        closest = min(scale, key=lambda x: abs(x - current))
        melody.append(closest)
    
    return melody

def compose_chords(root_notes, chord_type='triad'):
    """Generate chords from root notes"""
    chords = []
    
    for root in root_notes:
        if chord_type == 'triad':
            chord = [root, root + 4, root + 7]  # Major chord
        elif chord_type == 'seventh':
            chord = [root, root + 4, root + 7, root + 11]
        else:
            chord = [root]
        
        chords.append(chord)
    
    return chords

def compose_bass(chord_roots):
    """Generate a bass line based on chord roots"""
    bass = []
    for root in chord_roots:
        # Lower octave
        bass.append(root - 12)
    return bass

def create_midi_from_composition(melody, chords, bass, tempo=120, output_file='composition.mid', programs=None):
    """Create a MIDI file with ML predictions"""
    
    if programs is None:
        programs = {'melody': 0, 'chords': 0, 'bass': 32}
    
    print("Loading ML models...")
    rf_vel, rf_dur, scaler = load_models()
    if rf_vel:
        print("Models loaded\n")
    else:
        print("Models not found, using defaults\n")
    
    midi = MidiFile(ticks_per_beat=480)
    
    # TRACK 1: MELODY
    melody_track = MidiTrack()
    midi.tracks.append(melody_track)
    
    melody_track.append(MetaMessage('track_name', name='Melody'))
    melody_track.append(MetaMessage('set_tempo', tempo=int(60_000_000 / tempo)))
    melody_track.append(Message('program_change', program=programs['melody'], time=0))
    
    velocities, durations = predict_ml_params(melody, rf_vel, rf_dur, scaler)
    
    print(f"Melody: {len(melody)} notes")
    
    time = 0
    for i, (pitch, vel, dur) in enumerate(zip(melody, velocities, durations)):
        melody_track.append(Message('note_on', note=pitch, velocity=vel, time=time))
        melody_track.append(Message('note_off', note=pitch, velocity=0, time=dur))
        time = max(0, 480 - dur)  # Gap between notes, ensure non-negative
    
    chord_track = MidiTrack()
    midi.tracks.append(chord_track)
    
    chord_track.append(MetaMessage('track_name', name='Chords'))
    chord_track.append(Message('program_change', program=programs['chords'], time=0))
    
    all_chord_notes = [note for chord in chords for note in chord]
    chord_vels, chord_durs = predict_ml_params(all_chord_notes, rf_vel, rf_dur, scaler)
    
    print(f"Chords: {len(chords)} chords")
    
    idx = 0
    time = 0
    for chord in chords:
        for note in chord:
            vel = chord_vels[idx] if idx < len(chord_vels) else 70
            chord_track.append(Message('note_on', note=note, velocity=vel, time=time))
            time = 0
            idx += 1
        
        for note in chord:
            chord_track.append(Message('note_off', note=note, velocity=0, time=1920))
    
    bass_track = MidiTrack()
    midi.tracks.append(bass_track)
    
    bass_track.append(MetaMessage('track_name', name='Bass'))
    bass_track.append(Message('program_change', program=programs['bass'], time=0))
    
    bass_vels, bass_durs = predict_ml_params(bass, rf_vel, rf_dur, scaler)
    
    print(f"Bass: {len(bass)} notes")
    
    time = 0
    for pitch, vel, dur in zip(bass, bass_vels, bass_durs):
        # Ensure pitch is in valid MIDI range
        pitch = max(0, min(127, pitch))
        bass_track.append(Message('note_on', note=pitch, velocity=vel, time=time))
        bass_track.append(Message('note_off', note=pitch, velocity=0, time=1920))
        time = 0
    
    midi.save(output_file)
    print(f"\nSaved: {output_file}\n")

# Genre-based composition

def compose_genre(genre_key, num_bars=8, output_file=None):
    """Generate a composition in a specific genre"""
    
    if genre_key not in GENRES:
        print(f"Genre '{genre_key}' unknown. Available: {', '.join(GENRES.keys())}")
        return
    
    genre = GENRES[genre_key]
    
    if output_file is None:
        output_file = f'output/{genre_key}.mid'
    
    print(f"\nGenerating: {genre['name']}")
    
    # Melody - longer for extended duration
    num_notes = num_bars * 8
    start_pitch = sum(genre['melody_range']) // 2
    melody = compose_melody(
        genre['scale'], 
        num_notes=num_notes, 
        start_pitch=start_pitch,
        melody_steps=genre['melody_steps'],
        step_probs=genre['step_probs'],
        melody_range=genre['melody_range']
    )
    
    # Chords (1 per bar)
    chord_roots = [genre['scale'][i % len(genre['scale'])] for i in [0, 3, 4, 0]]
    # Extend chord progression to match num_bars
    chord_roots = (chord_roots * ((num_bars // 4) + 1))[:num_bars]
    chords = compose_chords(chord_roots, chord_type=genre['chord_type'])
    
    # Bass
    bass = compose_bass(chord_roots)
    
    # Create MIDI
    create_midi_from_composition(
        melody, chords, bass, 
        tempo=genre['tempo'], 
        output_file=output_file,
        programs=genre['program']
    )
    
    render_to_wav(output_file, genre_key)

def get_available_presets():
    """Get list of available VST presets"""
    preset_dir = Path("presets")
    if not preset_dir.exists():
        return []
    
    presets = []
    for preset_file in preset_dir.glob("*.vstpreset"):
        presets.append(preset_file)
    
    return sorted(presets)

def select_preset():
    """Let user choose a preset or use defaults"""
    presets = get_available_presets()
    
    if not presets:
        return None
    
    print("\nAvailable presets:")
    print("0. No preset (use plugin default)")
    for i, preset in enumerate(presets, 1):
        print(f"{i}. {preset.stem}")
    
    try:
        choice = input("\nChoose preset (0-{}) or Enter for default: ".format(len(presets))).strip()
        
        if not choice or choice == "0":
            return None
        
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            return presets[idx]
        else:
            return None
            
    except (ValueError, KeyboardInterrupt):
        return None

def render_to_wav(midi_file, genre_key, preset_file=None):
    """Convert MIDI file to WAV using the C++ MidiRenderer"""
    
    # Paths
    renderer_exe = Path("build/bin/Release/MidiRenderer.exe")
    vst_plugin = r"C:\Program Files\Common Files\VST3\Spitfire Audio\LABS.vst3"
    
    # Output WAV file (same name as MIDI but .wav)
    midi_path = Path(midi_file)
    wav_output = midi_path.parent / f"{midi_path.stem}.wav"
    
    if not renderer_exe.exists():
        print(f"MidiRenderer not found at {renderer_exe}")
        print(f"Run: cmake --build build --config Release")
        return False
    
    if not Path(vst_plugin).exists():
        print(f"VST plugin not found at {vst_plugin}")
        return False
    
    if preset_file is None:
        preset_file = select_preset()
    
    if preset_file:
        print(f"Rendering to audio: {wav_output.name}")
        print(f"Using: Spitfire LABS with preset '{preset_file.stem}'")
        preset_path = str(preset_file.absolute())
    else:
        print(f"Rendering to audio: {wav_output.name}")
        print(f"Using: Spitfire LABS (default sound)")
        preset_path = "-"
    
    try:
        # Call the renderer
        cmd = [
            str(renderer_exe.absolute()),
            str(midi_path.absolute()),
            preset_path,
            vst_plugin,
            str(wav_output.absolute())
        ]
        
        result = subprocess.run(cmd)
        
        if wav_output.exists() and wav_output.stat().st_size > 0:
            print(f"\nAudio rendered successfully: {wav_output}")
            return True
        else:
            print(f"\nRendering failed - output file not created")
            return False
            
    except Exception as e:
        print(f"Error during rendering: {e}")
        return False

# Main execution

def main():
    """Main menu for genre selection"""
    print("\n" + "="*60)
    print("AI MIDI Composer - Genre Selection")
    print("="*60 + "\n")
    
    print("Available Genres:")
    for i, (key, genre) in enumerate(GENRES.items(), 1):
        print(f"{i}. {genre['name']:20} | Tempo: {genre['tempo']:3} BPM")
    print(f"{len(GENRES)+1}. Generate all genres")
    print()
    
    try:
        choice = input("Choose genre (1-6) or Enter for all: ").strip()
        
        # Ask for duration in bars
        bars_input = input("Number of bars (default: 8, max: 32): ").strip()
        try:
            num_bars = int(bars_input) if bars_input else 8
            num_bars = max(4, min(32, num_bars))  # Clamp between 4 and 32
        except ValueError:
            num_bars = 8
        
        print(f"\nGenerating {num_bars} bars (~{num_bars * 4} seconds)...\n")
        
        if not choice:
            print("\nGenerating all genres...\n")
            for key in GENRES.keys():
                compose_genre(key, num_bars=num_bars)
        elif choice.isdigit():
            idx = int(choice) - 1
            genre_keys = list(GENRES.keys())
            
            if 0 <= idx < len(genre_keys):
                compose_genre(genre_keys[idx], num_bars=num_bars)
            elif idx == len(genre_keys):
                print("\nGenerating all genres...\n")
                for key in GENRES.keys():
                    compose_genre(key, num_bars=num_bars)
            else:
                print("Invalid choice")
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nCancelled by user")

if __name__ == '__main__':
    main()